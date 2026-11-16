# 道路观测轨迹分析

Python 3.11 标准库项目，无安装步骤。所有成本均为 Fraction，CLI 以规范有理数字符串输出。

运行完整业务回归：

```sh
python3.11 -B -m unittest discover -s tests -t . -v
python3.11 -B run.py demo samples.json
```

独立入口：

```sh
python3.11 -B run.py route samples.json
python3.11 -B run.py analyze samples.json
```

## 输入及公开 API

- `Graph.from_dict(data)`：data 含 nodes、roads 及可选 forbidden。节点 ID、道路 ID 均为各自范围内唯一的非空字符串。道路对象含 id、tail、head、length，端点必须已声明；允许平行道路、自环和孤立节点。length 严格为正。
- forbidden 是若干长度为 2 或 3 的连续有向道路 ID 数组，必须首尾相接。匹配实际道路进入事件，初始行程没有未观测的先前历史。
- `graph.position({"node": id})` 或 `graph.position({"road": id, "offset": value})` 返回 Position。offset 是从道路起点量起的实际距离，范围为 [0,length]。0 与 length 规范化为端点节点；端点本身不声明进入该道路。
- `read_observations(graph, rows)` 返回 Observation 元组。每行含 time、candidates；时间非降序，允许相同时间。每个候选含 id、position、penalty；同一观测内 ID 唯一，penalty 非负。至少一个观测，空候选集合合法但不可达。
- `routing.shortest_route(graph, source, target)` 返回合法最短 Route，或 None。Route 含 distance、spans、history，roads 属性逐个列出道路发生；同成本任一解合法。
- `trajectory.analyze(graph, observations)` 返回全局最小 Analysis，或 None。Analysis 含 selected、distance、penalty、spans，提供 roads、total 属性。selected 按观测顺序记录候选 ID；total = 实际行驶长度 + 所选候选 penalty 之和。
- Python 调用也可构造 `Candidate(id, position, penalty)`、`Observation(time, candidates)`、`Position(node=...)`。入口仍验证并规范化数据。graph.position 是构造道路内部位置的推荐方式。

有理数接受整数、Fraction 或有理数/十进制字符串；布尔和浮点数拒绝。JSON 使用整数或字符串。时间仅给出观测顺序，当前不对速度或耗时建模。对象字段严格校验，非法值抛 ValueError；调用方不得修改图属性。

## 行程及结果语义

观测只接受一个候选，不重启行程。进入道路时推进连续道路历史；在同一道路内部前进、在同一位置重复观测不会再进入道路，也不会清空历史。不同道路历史即使在相同候选处也可能具有不同延续资格，选路与轨迹分析必须共同遵守限制。

spans 是有序的 `Span(road, start, end)` 元组，均沿有向道路前进。道路内部相邻片段合为一次道路发生，节点处再次进入同一道路（包括自环）保留为新的发生。因此 roads 可能连续出现同名道路。每段长度为 end-start，distance 为其和。

从道路内部开始意味着已经占据该道路，并将其纳入限制历史。若整个行程静止，其唯一 span 为该道路的零长度 span；若后来前进则与这段初始占据合并。除此以外 span 均为正长度。从节点开始并静止则 roads、spans 为空。道路内部不能后退；需要沿有向图绕回才能抵达更小 offset。

独立选路以未知此前历史开始。轨迹分析的道路限制跨所有观测持续有效，候选选择、完整路径与成本必须对应同一全局最优解。无合法连接返回不可达。同成本无需固定道路或候选顺序。

## CLI

输入为 UTF-8 JSON 对象：graph 必需；route 读取 source/target，analyze 读取 observations。允许一个文件同时保存两种请求；samples.json 即为有限样例。demo 调用两个真实入口并输出其结果。

成功输出 status=ok、roads、spans、distance、total；分析额外输出 selected、penalty。不可达仅输出 status=unreachable，退出 0。非法业务输入、坏 JSON 或文件读取错误输出 status=error 和 message，退出 2。命令行参数错误由 argparse 返回 2。

样例的独立路线和完整轨迹均为 d、b、c，距离 4，偏差成本 0。内部观测及重复观测不增加道路进入事件。

完整套件覆盖独立查询、跨观测限制、候选联合选择、内部位置、重复观测、自环、不可达、同成本及输入拒绝。测试参照独立细分道路为观测层状态图，显式枚举有限可达状态后用 Bellman-Ford 核对成本，并独立重放结果道路与候选经过关系。实现面向有限离线输入，当前逐个连接执行完整状态搜索并保存路径，不承诺大型路网性能。
