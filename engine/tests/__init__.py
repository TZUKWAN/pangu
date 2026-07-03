"""测试包。

测试分两类：
- test_*_logic.py：纯逻辑测试（用 mock 数据，不依赖网络），快、稳定
- test_*_live.py：连真实 akshare 的冒烟测试（需网络，CI 可跳过）

跑全部：pytest engine/tests -v
只跑逻辑：pytest engine/tests -v -k "not live"
"""
