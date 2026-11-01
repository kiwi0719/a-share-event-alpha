"""中标 / 重大合同公告：标题筛选与正文参数抽取（规则版，作为 NLP 抽取的基线）。

抽取两个参数：
- amount_cny：合同或中标金额（元），取正文中第一个与“金额/价款/中标价/合同价”关联的数字；
- revenue_ratio：公告自述的“占公司最近一个会计年度经审计营业收入的 X%”。

只做事实抽取，不判断利好利空。抽不到时返回 None，交给后续人工或模型。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

POSITIVE_TITLE = re.compile(r"中标|签订.{0,20}合同|重大合同|签署.{0,20}(合同|协议)|项目.{0,6}(成交|中选)|预中标|候选人")
EXCLUDE_TITLE = re.compile(r"进展|补充|更正|更新|终止|解除|取消|延期|摘要|英文|提示性.*?风险|法律意见|核查意见|关联交易|担保|借款|贷款|理财|募集|回购|股权转让|增资|租赁")
CANDIDATE_TITLE = re.compile(r"候选人|预中标|拟中标|中标候选")

_NUM = r"([0-9][0-9,，]*(?:\.[0-9]+)?)"
_UNIT = r"\s*(亿元|万元|元|亿|万)"
AMOUNT = re.compile(
    r"(?:中标金额|中标价|合同金额|合同总金额|合同价款|合同总价|成交金额|成交价|交易金额|总金额|总价款|总价|金额)"
    r"[^0-9。；;\n]{0,30}?" + _NUM + _UNIT
)
AMOUNT_FALLBACK = re.compile(r"(?:人民币|约)\s*" + _NUM + _UNIT)
RATIO = re.compile(
    r"占[^。；;\n]{0,40}?营业(?:总)?收入[^0-9。；;\n]{0,15}?" + r"([0-9]+(?:\.[0-9]+)?)\s*[%％]"
)
FOREIGN = re.compile(r"美元|欧元|港元|港币|日元|英镑|卢布|雷亚尔|迪拉姆|里亚尔|澳元|加元")

_SCALE = {"亿元": 1e8, "亿": 1e8, "万元": 1e4, "万": 1e4, "元": 1.0}


@dataclass
class ContractFacts:
    amount_cny: float | None
    revenue_ratio: float | None  # 0.12 表示 12%
    is_candidate: bool  # 中标候选人/预中标，尚未确定
    foreign_currency: bool


def classify_title(title: str) -> bool:
    return bool(POSITIVE_TITLE.search(title)) and not EXCLUDE_TITLE.search(title)


def _to_num(s: str) -> float:
    return float(s.replace(",", "").replace("，", ""))


def extract(title: str, text: str) -> ContractFacts:
    text = re.sub(r"[ \t　]+", "", text or "")
    amt = None
    m = AMOUNT.search(text) or AMOUNT_FALLBACK.search(text)
    if m:
        amt = _to_num(m.group(1)) * _SCALE[m.group(2)]
    ratio = None
    r = RATIO.search(text)
    if r:
        v = float(r.group(1))
        if 0 < v < 10000:
            ratio = v / 100.0
    window = text[max(0, (m.start() if m else 0) - 20) : (m.end() if m else 0) + 20] if m else ""
    return ContractFacts(
        amount_cny=amt,
        revenue_ratio=ratio,
        is_candidate=bool(CANDIDATE_TITLE.search(title)),
        foreign_currency=bool(FOREIGN.search(window)),
    )
