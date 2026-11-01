from event_alpha.events.contracts import classify_title, extract


def test_classify_title():
    assert classify_title("某某股份:关于签订日常经营重大合同的公告")
    assert classify_title("某某股份:关于项目中标的公告")
    assert not classify_title("某某股份:关于重大合同的进展公告")
    assert not classify_title("某某股份:关于签订借款合同的公告")


def test_extract_amount_and_ratio():
    t = "合同金额：人民币 1,234.56 万元（含税），占公司 2023 年度经审计营业收入的 12.5%。"
    f = extract("关于签订重大合同的公告", t)
    assert abs(f.amount_cny - 12345600) < 1
    assert abs(f.revenue_ratio - 0.125) < 1e-9


def test_extract_yi_and_candidate():
    f = extract("关于项目预中标的提示性公告", "中标价为 3.2 亿元。")
    assert f.amount_cny == 3.2e8 and f.is_candidate and f.revenue_ratio is None


def test_foreign_currency_flag():
    f = extract("关于签订重大合同的公告", "合同金额约 5,000 万美元")
    assert f.foreign_currency
