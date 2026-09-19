"""程序化交易合规门禁测试：状态机、LiveGate 逐项失败、信息表导出。"""
from __future__ import annotations

import json

import pytest

from engine.contracts import ComplianceState, ExecutionMode, StrategyStatus
from engine.compliance.program_trading import (
    ComplianceManager,
    IllegalComplianceTransition,
    LiveGate,
)


@pytest.fixture()
def mgr(tmp_path):
    return ComplianceManager(db_path=str(tmp_path / "compliance.db"),
                             audit_dir=str(tmp_path / "audit"))


class TestComplianceStateMachine:
    def test_initial_state_unknown(self, mgr):
        assert mgr.state() == ComplianceState.UNKNOWN

    def test_happy_path_report_to_confirmed(self, mgr):
        mgr.set_state(ComplianceState.REPORT_REQUIRED, "op", "办法落地，需报告")
        mgr.set_state(ComplianceState.SUBMITTED, "op", "报告已提交 2026-09-15")
        mgr.set_state(ComplianceState.CONFIRMED, "op", "回执编号 X")
        assert mgr.state() == ComplianceState.CONFIRMED
        assert [h["to_state"] for h in mgr.history()] == \
            ["REPORT_REQUIRED", "SUBMITTED", "CONFIRMED"]

    def test_unknown_direct_not_required_confirmed(self, mgr):
        mgr.set_state(ComplianceState.NOT_REQUIRED_CONFIRMED, "op", "账户认定无需报告")
        assert mgr.state() == ComplianceState.NOT_REQUIRED_CONFIRMED

    def test_illegal_jump_unknown_to_confirmed(self, mgr):
        with pytest.raises(IllegalComplianceTransition):
            mgr.set_state(ComplianceState.CONFIRMED, "op", "跳步")

    def test_illegal_not_required_is_terminal(self, mgr):
        mgr.set_state(ComplianceState.NOT_REQUIRED_CONFIRMED, "op", "认定")
        with pytest.raises(IllegalComplianceTransition):
            mgr.set_state(ComplianceState.CONFIRMED, "op", "不得跳")

    def test_confirmed_only_to_expired(self, mgr):
        mgr.set_state(ComplianceState.REPORT_REQUIRED, "op", "e")
        mgr.set_state(ComplianceState.SUBMITTED, "op", "e")
        mgr.set_state(ComplianceState.CONFIRMED, "op", "e")
        with pytest.raises(IllegalComplianceTransition):
            mgr.set_state(ComplianceState.REPORT_REQUIRED, "op", "不得跳")
        mgr.set_state(ComplianceState.EXPIRED_OR_CHANGED, "op", "要素变更")
        mgr.set_state(ComplianceState.REPORT_REQUIRED, "op", "重报")
        assert mgr.state() == ComplianceState.REPORT_REQUIRED

    def test_state_persisted_across_instances(self, tmp_path):
        db = str(tmp_path / "c.db")
        m1 = ComplianceManager(db_path=db, audit_dir=str(tmp_path / "a"))
        m1.set_state(ComplianceState.REPORT_REQUIRED, "op", "e")
        m2 = ComplianceManager(db_path=db, audit_dir=str(tmp_path / "a"))
        assert m2.state() == ComplianceState.REPORT_REQUIRED

    def test_can_enable_live(self, mgr):
        ok, _ = mgr.can_enable_live()
        assert ok is False
        mgr.set_state(ComplianceState.NOT_REQUIRED_CONFIRMED, "op", "认定")
        ok, reason = mgr.can_enable_live()
        assert ok is True and "NOT_REQUIRED_CONFIRMED" in reason

    def test_audit_jsonl_appended(self, mgr, tmp_path):
        mgr.set_state(ComplianceState.SUBMITTED, "op", "提交")
        log = tmp_path / "audit" / "compliance_log.jsonl"
        lines = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
        assert lines[-1]["event"] == "state_change"
        assert lines[-1]["to"] == "SUBMITTED" and lines[-1]["operator"] == "op"


class TestTradingSoftwareInfoExport:
    def test_export_writes_report(self, mgr, tmp_path):
        info = mgr.export_trading_software_info(
            rate_limit_cfg={"max_per_day": 200, "min_interval_seconds": 2},
            kill_switch_info={"risk_controls": "七道盘前检查", "emergency_plan": "一键熔断"})
        assert info["软件名称"] == "Pangu"
        assert info["策略类型"] == "低中频A股现货"
        assert info["指令生成方式"] == "策略信号→组合→OMS限价单"
        assert "200" in info["最大预计下单频率"] and "2" in info["最大预计下单频率"]
        assert info["风控摘要"] == "七道盘前检查" and info["应急方案"] == "一键熔断"
        assert info["版本"] and info["版本"] != "unknown"  # git sha 可得
        path = tmp_path / "audit" / "program_trading_report.json"
        assert json.loads(path.read_text(encoding="utf-8")) == info


class TestLiveGate:
    ALL_OK = dict(strategy_status=StrategyStatus.LIVE, execution_mode=ExecutionMode.LIVE,
                  compliance_ok=True, paper_passed=True, shadow_passed=True, broker_ok=True,
                  reconcile_ok=True, kill_switch_ready=True)

    def evaluate(self, **overrides):
        kw = {**self.ALL_OK, **overrides}
        return LiveGate.evaluate(**kw)

    def test_all_green_approves(self):
        res = self.evaluate()
        assert res.approved is True and res.failed_checks == []
        assert res.checked_at

    def test_fails_without_compliance(self):
        res = self.evaluate(compliance_ok=False)
        assert res.approved is False
        assert "compliance_not_confirmed" in res.failed_checks

    def test_fails_without_paper(self):
        res = self.evaluate(paper_passed=False)
        assert res.failed_checks == ["paper_not_passed"]

    def test_fails_without_shadow(self):
        res = self.evaluate(shadow_passed=False)
        assert res.failed_checks == ["shadow_not_passed"]

    def test_fails_without_broker(self):
        res = self.evaluate(broker_ok=False)
        assert res.failed_checks == ["broker_not_ok"]

    def test_fails_without_reconcile(self):
        res = self.evaluate(reconcile_ok=False)
        assert res.failed_checks == ["reconcile_not_ok"]

    def test_fails_without_kill_switch(self):
        res = self.evaluate(kill_switch_ready=False)
        assert res.failed_checks == ["kill_switch_not_ready"]

    def test_fails_wrong_execution_mode(self):
        res = self.evaluate(execution_mode=ExecutionMode.PAPER)
        assert res.approved is False
        assert any(c.startswith("execution_mode_not_live") for c in res.failed_checks)

    def test_fails_strategy_not_live_capable(self):
        res = self.evaluate(strategy_status=StrategyStatus.PAPER)
        assert res.approved is False
        assert any(c.startswith("strategy_status_not_live_capable") for c in res.failed_checks)

    def test_limited_live_status_is_live_capable(self):
        res = self.evaluate(strategy_status=StrategyStatus.LIMITED_LIVE)
        assert res.approved is True

    def test_fails_closed_on_multiple_missing(self):
        res = self.evaluate(compliance_ok=False, paper_passed=False, broker_ok=False)
        assert res.approved is False and len(res.failed_checks) == 3
