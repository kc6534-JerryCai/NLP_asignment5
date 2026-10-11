from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.graphics.shapes import Drawing, Line, Rect, String


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "assignment5_guardian_topics_and_markets_report.pdf"

NAVY = colors.HexColor("#14233A")
BLUE = colors.HexColor("#294A6D")
MID = colors.HexColor("#64798E")
LINE = colors.HexColor("#9DAFC0")
PALE = colors.HexColor("#EEF2F5")
PALE_BLUE = colors.HexColor("#E8EFF5")
WHITE = colors.white
BLACK = colors.HexColor("#1F252B")
RED = colors.HexColor("#9E2A2B")
GREEN = colors.HexColor("#2D6A4F")


def fmt_num(x, digits=3, na="--"):
    if pd.isna(x):
        return na
    return f"{float(x):.{digits}f}"


def fmt_q(x):
    if pd.isna(x):
        return "--"
    x = float(x)
    return f"{x:.2e}" if x < 0.001 else f"{x:.3f}"


def safe_bool(x):
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"true", "1", "yes"}


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    "ReportTitle", parent=styles["Title"], fontName="Times-Bold", fontSize=22,
    leading=26, textColor=NAVY, alignment=TA_CENTER, spaceAfter=14,
))
styles.add(ParagraphStyle(
    "ReportSubtitle", parent=styles["Normal"], fontName="Times-Roman", fontSize=11,
    leading=15, textColor=MID, alignment=TA_CENTER, spaceAfter=6,
))
styles.add(ParagraphStyle(
    "RepoLink", parent=styles["Normal"], fontName="Times-Roman", fontSize=8.5,
    leading=11, textColor=BLUE, alignment=TA_CENTER, spaceAfter=5,
))
styles.add(ParagraphStyle(
    "H1x", parent=styles["Heading1"], fontName="Times-Bold", fontSize=14,
    leading=17, textColor=NAVY, spaceBefore=8, spaceAfter=7,
))
styles.add(ParagraphStyle(
    "H2x", parent=styles["Heading2"], fontName="Times-Bold", fontSize=11,
    leading=14, textColor=BLUE, spaceBefore=6, spaceAfter=5,
))
styles.add(ParagraphStyle(
    "Bodyx", parent=styles["BodyText"], fontName="Times-Roman", fontSize=9.2,
    leading=12.4, textColor=BLACK, alignment=TA_JUSTIFY, spaceAfter=6,
))
styles.add(ParagraphStyle(
    "Smallx", parent=styles["BodyText"], fontName="Times-Roman", fontSize=7.6,
    leading=9.7, textColor=BLACK, alignment=TA_LEFT, spaceAfter=3,
))
styles.add(ParagraphStyle(
    "Captionx", parent=styles["BodyText"], fontName="Times-Italic", fontSize=7.7,
    leading=9.8, textColor=MID, alignment=TA_JUSTIFY, spaceBefore=3, spaceAfter=7,
))
styles.add(ParagraphStyle(
    "BoxHead", parent=styles["BodyText"], fontName="Times-Bold", fontSize=9.4,
    leading=12, textColor=NAVY, alignment=TA_LEFT, spaceAfter=4,
))
styles.add(ParagraphStyle(
    "BoxBody", parent=styles["BodyText"], fontName="Times-Roman", fontSize=8.6,
    leading=11.3, textColor=BLACK, alignment=TA_JUSTIFY,
))


def P(text, style="Bodyx"):
    return Paragraph(text, styles[style])


def cell(text, bold=False, size=6.5, color=BLACK, align=TA_LEFT):
    return Paragraph(
        str(text),
        ParagraphStyle(
            name=f"cell_{bold}_{size}_{align}", fontName="Times-Bold" if bold else "Times-Roman",
            fontSize=size, leading=size + 1.6, textColor=color, alignment=align,
        ),
    )


def academic_table(headers, rows, widths, font_size=6.5, repeat_rows=1, right_cols=None):
    right_cols = set(right_cols or [])
    data = [[cell(h, True, font_size, WHITE, TA_CENTER) for h in headers]]
    for row in rows:
        rr = []
        for j, value in enumerate(row):
            rr.append(cell(value, False, font_size, BLACK, TA_RIGHT if j in right_cols else TA_LEFT))
        data.append(rr)
    t = Table(data, colWidths=widths, repeatRows=repeat_rows, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3.4),
        ("TOPPADDING", (0, 0), (-1, -1), 3.0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.0),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            commands.append(("BACKGROUND", (0, i), (-1, i), PALE))
    t.setStyle(TableStyle(commands))
    return t


def callout(title, body):
    box = Table([[P(title, "BoxHead"), P(body, "BoxBody")]], colWidths=[1.35 * inch, 5.35 * inch])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
        ("BOX", (0, 0), (-1, -1), 0.8, BLUE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return box


def figure(path, caption, width=6.75 * inch, height=None):
    path = ROOT / path
    im = Image(str(path))
    if height is None:
        ratio = im.imageHeight / im.imageWidth
        height = width * ratio
    im.drawWidth = width
    im.drawHeight = height
    return KeepTogether([im, P(caption, "Captionx")])


def stock_effect_chart(rows):
    """Compact vector chart for the same representative stocks shown in Table 8."""
    d = Drawing(486, 315)
    d.add(String(0, 298, "Representative strict stock effects", fontName="Times-Bold", fontSize=11, fillColor=NAVY))
    d.add(String(0, 284, "Market-adjusted basis points per one-standard-deviation signal increase", fontName="Times-Roman", fontSize=7.5, fillColor=MID))
    plot_left, plot_right = 170, 455
    x_min, x_max = -140.0, 90.0

    def xmap(v):
        return plot_left + (float(v) - x_min) / (x_max - x_min) * (plot_right - plot_left)

    for tick in [-120, -60, 0, 60]:
        x = xmap(tick)
        d.add(Line(x, 26, x, 270, strokeColor=LINE if tick else MID, strokeWidth=0.45 if tick else 0.8))
        d.add(String(x, 14, f"{tick:+d}", fontName="Times-Roman", fontSize=6.8, textAnchor="middle", fillColor=MID))

    y0, step = 258, 19
    idx_short = {"topic_sentiment": "sent.", "democrat_stance": "Dem.", "republican_stance": "Rep."}
    for i, (_, r) in enumerate(rows.iterrows()):
        y = y0 - i * step
        if i % 2:
            d.add(Rect(0, y - 7.5, 480, 16, fillColor=PALE, strokeColor=None))
        label = f"{r.ticker}   T{int(r.topic_id)} {idx_short[r.index_type]}"
        d.add(String(2, y - 2.8, label, fontName="Times-Roman", fontSize=7.5, fillColor=BLACK))
        val = float(r.standardized_effect_bps)
        zx, vx = xmap(0), xmap(val)
        left, width = min(zx, vx), abs(vx - zx)
        d.add(Rect(left, y - 5.5, max(width, 1), 9, fillColor=GREEN if val > 0 else RED, strokeColor=None))
        anchor = "start" if val >= 0 else "end"
        text_x = min(vx + 4, 480) if val >= 0 else max(vx - 4, 174)
        d.add(String(text_x, y - 3.0, f"{val:+.1f}", fontName="Times-Bold", fontSize=6.8, textAnchor=anchor, fillColor=BLACK))
    d.add(String((plot_left + plot_right) / 2, 0, "Standardized effect (basis points)", fontName="Times-Italic", fontSize=7, textAnchor="middle", fillColor=MID))
    return KeepTogether([
        d,
        P("Figure 6. The same representative FDR-and-bootstrap-supported stock findings listed in Table 8. Positive and negative bars indicate the sign of the within-sample association; they are not long/short recommendations or causal effects.", "Captionx"),
    ])


class ReportDoc(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(
            filename, pagesize=letter, leftMargin=0.72 * inch, rightMargin=0.72 * inch,
            topMargin=0.68 * inch, bottomMargin=0.62 * inch,
            title="Guardian Midterm Policy Topics and Market Associations",
            author="Kuanhong Cai",
            subject="Assignment 5 research report",
        )
        frame_cover = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="cover")
        frame_body = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="body")
        self.addPageTemplates([
            PageTemplate(id="Cover", frames=frame_cover, onPage=self.cover_page),
            PageTemplate(id="Body", frames=frame_body, onPage=self.body_page),
        ])

    @staticmethod
    def cover_page(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.5)
        canvas.line(doc.leftMargin, 0.49 * inch, letter[0] - doc.rightMargin, 0.49 * inch)
        canvas.setFont("Times-Roman", 7.5)
        canvas.setFillColor(MID)
        canvas.drawCentredString(letter[0] / 2, 0.31 * inch, "Guardian News, Election Markets, and U.S. Equities | October 2026")
        canvas.restoreState()

    @staticmethod
    def body_page(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.45)
        canvas.line(doc.leftMargin, letter[1] - 0.43 * inch, letter[0] - doc.rightMargin, letter[1] - 0.43 * inch)
        canvas.setFont("Times-Italic", 7.5)
        canvas.setFillColor(MID)
        canvas.drawString(doc.leftMargin, letter[1] - 0.31 * inch, "U.S. Midterm Policy Topics and Market Associations")
        canvas.drawRightString(letter[0] - doc.rightMargin, letter[1] - 0.31 * inch, "Kuanhong Cai")
        canvas.line(doc.leftMargin, 0.49 * inch, letter[0] - doc.rightMargin, 0.49 * inch)
        canvas.setFont("Times-Roman", 8)
        canvas.drawCentredString(letter[0] / 2, 0.31 * inch, str(doc.page))
        canvas.restoreState()


def load_inputs():
    topics = pd.read_csv(ROOT / "data/lda_policy/policy_topic_labels.csv")
    keywords = pd.read_csv(ROOT / "data/lda_policy/policy_topic_keywords.csv")
    attention = pd.read_csv(ROOT / "data/lda_policy/policy_overall_topic_attention.csv")
    comparison = pd.read_csv(ROOT / "data/lda_policy/policy_model_comparison.csv")
    sweep_corr = pd.read_csv(ROOT / "data/sweep_analysis/correlation_results.csv")
    sweep_oos = pd.read_csv(ROOT / "data/sweep_analysis/walk_forward_metrics.csv")
    stock_fdr = pd.read_csv(ROOT / "data/equity_sentiment/fdr_significant_results.csv")
    stock_strict = pd.read_csv(ROOT / "data/equity_sentiment/fdr_significant_bootstrap_supported.csv")
    stock_all = pd.read_csv(ROOT / "data/equity_sentiment/main_univariate_tests.csv")
    baskets = pd.read_csv(ROOT / "data/equity_sentiment/midterm_basket/midterm_basket_tests.csv")
    with open(ROOT / "data/equity_sentiment/analysis_summary.json", encoding="utf-8") as f:
        stock_summary = json.load(f)
    with open(ROOT / "data/equity_sentiment/midterm_basket/midterm_basket_summary.json", encoding="utf-8") as f:
        basket_summary = json.load(f)
    return topics, keywords, attention, comparison, sweep_corr, sweep_oos, stock_fdr, stock_strict, stock_all, baskets, stock_summary, basket_summary


def build_story():
    (
        topics, keywords, attention, comparison, sweep_corr, sweep_oos, stock_fdr,
        stock_strict, stock_all, baskets, stock_summary, basket_summary,
    ) = load_inputs()
    topic_map = topics.set_index("topic_id")["english_name"].to_dict()
    att_col = "overall_attention_share"

    story = []
    # Cover
    story += [Spacer(1, 0.42 * inch), P("U.S. Midterm Policy Topics in Guardian News<br/>and Their Market Associations", "ReportTitle")]
    story += [P("Topic Modeling, Democratic Sweep Prediction-Market Tests, and Equity Evidence", "ReportSubtitle")]
    story += [
        Spacer(1, 0.17 * inch),
        P("Kuanhong Cai", "ReportSubtitle"),
        P("New York University", "ReportSubtitle"),
        P("October 2026", "ReportSubtitle"),
        P('<link href="https://github.com/kc6534-JerryCai/NLP_asignment5"><u>github.com/kc6534-JerryCai/NLP_asignment5</u></link>', "RepoLink"),
        Spacer(1, 0.16 * inch),
    ]
    abstract = (
        "This report studies substantive policy coverage surrounding the 2026 U.S. midterm elections in 1,216 Guardian articles collected from July 10 through October 9, 2026. "
        "A policy-focused screening stage retained 227 articles, of which 226 passed text-length requirements. A six-topic LDA model identifies technology governance, infrastructure, trade and rights, Iran and energy security, immigration and household economics, and defense. "
        "DeepSeek-based article annotations separate topic sentiment from Democratic and Republican stance. The resulting time series are compared with Polymarket's joint Democratic Senate-and-House sweep contract and with market-adjusted U.S. equity returns. "
        "The Democratic Sweep tests produce no association that survives multiplicity adjustment. Equity regressions identify broad statistically significant relationships across individual stocks and topic-specific baskets, with economically meaningful effect sizes in several policy signals. "
        "The results indicate that the policy-news measures contain market-relevant information during the study window, while the reported associations are interpreted without causal claims."
    )
    abox = Table([[P("ABSTRACT", "BoxHead")], [P(abstract, "BoxBody")]], colWidths=[6.65 * inch])
    abox.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.9, BLUE), ("BACKGROUND", (0, 0), (-1, 0), PALE_BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story += [abox, Spacer(1, 0.14 * inch), P("<b>Keywords:</b> topic modeling; LDA; news sentiment; U.S. midterms; Polymarket; market-adjusted returns; multiple testing", "Smallx")]
    story += [Spacer(1, 0.22 * inch), callout(
        "Principal finding",
        "The topic model produces interpretable policy clusters and useful descriptive time series, while the market evidence differs by outcome. Democratic Sweep correlations are not statistically reliable after BH adjustment. In contrast, the equity results show broad and economically meaningful associations at both the individual-stock and dedicated-basket levels.",
    )]
    story += [NextPageTemplate("Body"), PageBreak()]

    # 1 Introduction and data
    story += [P("1. Research Design and Data", "H1x")]
    story += [P(
        "The project separates three quantities that are often conflated: <i>attention</i> (how much an article loads on a topic), <i>topic sentiment</i> (the evaluative tone toward the policy condition or proposal), and <i>party stance</i> (support for or criticism of Democrats or Republicans within the topic). Attention is obtained from LDA probabilities; sentiment and stance are article-level language-model annotations. Missing expression is never recoded as neutral, and missing news is never filled with zero.",
    )]
    overview_rows = [
        ["Guardian corpus", "1,216 English-language articles", "2026-07-10 to 2026-10-09 (New York dates)"],
        ["Policy LDA", "226 valid policy documents", "One full article body per document; K=6"],
        ["Article annotations", "1,356 article-topic rows", "226 articles x 6 topics; no failed articles"],
        ["Democratic Sweep", "2,205 hourly observations", "92 daily closing proxies; 14 appended out-of-range records removed"],
        ["Equities", "999 downloaded / 998 testable", "Market-adjusted open-to-close return versus SPY"],
        ["Research window", "92 natural days", "Calendar alignment preserves weekends and missingness"],
    ]
    story += [P("Table 1. Core datasets", "H2x"), academic_table(["Component", "Scale", "Definition"], overview_rows, [1.25*inch, 1.55*inch, 3.9*inch], 7.2)]
    story += [P("1.1 Policy-scope sample", "H2x")]
    story += [P(
        "The original screening classified 641 articles as broadly relevant, 424 as uncertain, and 151 as irrelevant. A second policy-centrality screen produced 227 policy-substance articles, 103 mixed policy/campaign articles, 249 campaign-process articles, 486 uncertain articles, and 151 excluded articles. These five categories exhaust the 1,216-article corpus. The main model intentionally uses only policy-substance articles; one document fell below the 30-token threshold, leaving 226 LDA documents. Mixed and uncertain records remain available for sensitivity analysis but are not folded into the headline model.",
    )]
    story += [callout("Interpretation boundary", "All attention and sentiment estimates describe the Guardian policy-news sample. They are not measures of U.S. voter preferences, approval, voting intention, or causal media effects.")]
    story += [P("1.2 Market and equity outcomes", "H2x")]
    story += [P(
        "The prediction-market outcome is the verified Polymarket contract <i>2026 Balance of Power: D Senate, D House</i> (group label: Democrats Sweep; market ID 562828). The analysis uses the Yes token, downloaded in at-most-seven-day blocks at hourly fidelity. Each New York day is represented by its last valid observation; no cross-day forward fill is used. Equity outcomes are split-adjusted open-to-close stock returns minus the contemporaneous SPY return, which targets stock-specific movement rather than broad-market direction.",
    )]
    story += [PageBreak()]

    # 2 NLP methods
    story += [P("2. NLP Method: Policy-Focused Topic Modeling", "H1x")]
    story += [P("2.1 Text preparation", "H2x")]
    story += [P(
        "Each complete article body is one document; titles are retained for interpretation but are not appended to the body. The pipeline removes HTML, URLs, abnormal whitespace, and source-template fragments while preserving the original article separately. spaCy (<i>en_core_web_sm</i>) lowercases, tokenizes, and lemmatizes English text. Stopwords, pure punctuation, pure numbers, and clearly non-informative tokens are removed. Gensim's phrase model detects recurring bigrams such as <i>service_member</i> and <i>strait_hormuz</i>. The acronym <i>AI</i> is explicitly retained. Election-mechanics and high-frequency political context terms are removed only from LDA tokens after inspection showed that they merged otherwise distinct policy clusters; the source text remains unchanged.",
    )]
    prep_rows = [
        ["Document unit", "One complete article body"],
        ["Minimum usable length", "30 processed tokens; 1 article excluded"],
        ["Phrase model", "min_count=4; threshold=8.0"],
        ["Dictionary filter", "no_below=4; no_above=0.75"],
        ["Vocabulary", "10,943 terms before filtering; 3,513 after filtering"],
        ["LDA input", "Gensim Dictionary + doc2bow term frequencies; no TF-IDF"],
        ["Estimator", "Gensim LdaModel; passes=10; iterations=100; alpha=auto; eta=auto"],
    ]
    story += [academic_table(["Element", "Implementation"], prep_rows, [1.8*inch, 4.9*inch], 7.3)]
    story += [P("2.2 Candidate models and topic-number selection", "H2x")]
    story += [P(
        "Candidate models K=4 through K=8 were trained with random state 42. Selection considered c_v coherence, cross-topic keyword overlap, representative articles, topic size, and interpretability; coherence alone did not determine K. The K=6 model has the highest coherence in the candidate set and preserves a distinct AI topic while separating energy-security, defense, infrastructure, immigration/economics, and trade/rights coverage. Stability checks refit K=6 with seeds 7 and 99 and matched topics by their word distributions rather than topic IDs.",
    )]
    comp_rows = []
    for _, r in comparison.iterrows():
        comp_rows.append([
            int(r.num_topics), fmt_num(r.overall_c_v_coherence, 3), fmt_num(r.top15_unique_term_ratio, 3),
            fmt_num(r.mean_pairwise_top15_jaccard, 3), fmt_num(r.mean_dominant_topic_probability, 3),
            "Selected" if safe_bool(r.recommended) else "Candidate",
        ])
    story += [P("Table 2. LDA candidate comparison", "H2x"), academic_table(
        ["K", "c_v", "Unique top-15", "Mean overlap", "Mean dominant p", "Decision"],
        comp_rows, [0.45*inch,0.65*inch,1.0*inch,1.0*inch,1.15*inch,1.0*inch], 6.9, right_cols={0,1,2,3,4}
    )]
    story += [P(
        "The selected model's matched-seed mean full-vocabulary cosine similarity is 0.728 and its mean top-15 Jaccard overlap is 0.232. These values support broad reproducibility but also show that exact word rankings are not invariant. Topic names are therefore documented human interpretations, not facts emitted directly by LDA.", "Smallx")]
    story += [PageBreak()]

    # 3 topics
    story += [P("3. Topic Model Results", "H1x")]
    topic_rows = []
    for tid in range(6):
        tr = topics.loc[topics.topic_id == tid].iloc[0]
        share = float(attention.loc[attention.topic_id == tid, att_col].iloc[0])
        kws = ", ".join(keywords.loc[(keywords.topic_id == tid) & (keywords["rank"] <= 8), "keyword"].astype(str))
        topic_rows.append([f"T{tid}", tr.english_name, f"{share:.1%}", kws])
    story += [P("Table 3. Final policy topics and conditional attention shares", "H2x"), academic_table(
        ["Topic", "Provisional interpretation", "Share", "Leading terms"], topic_rows,
        [0.42*inch, 2.42*inch, 0.58*inch, 3.28*inch], 6.4, right_cols={2}
    )]
    story += [P(
        "T2 and T3 jointly account for 47.4% of probability-weighted attention. T2 is the least internally coherent topic because it combines trade, public-health, and legal-rights material; T4 is also broad, combining immigration enforcement with public funding and household economics. These mixed clusters should be interpreted with their subissue annotations rather than as single narrow policy concepts.",
    )]
    story += [figure("figures/lda_policy/policy_overall_attention.png", "Figure 1. Overall topic attention in the 226-document policy sample. Shares are averages of complete document-topic probability vectors and sum to one; they are conditional on inclusion in the policy sample.", width=6.35*inch)]
    story += [callout("Why energy and AI appear", "The policy-only design removes campaign-process language that dominated the earlier all-discourse model. This allows T0 (AI and digital governance) and T3 (Iran, oil, and energy security) to emerge as distinct clusters rather than being absorbed by generic election terminology.")]
    story += [PageBreak()]

    # 4 time series
    story += [P("4. Policy Attention and Sentiment over Time", "H1x")]
    story += [P(
        "Article-level sentiment is labeled positive (+1), negative (-1), neutral (0), mixed (0), or unclear/missing. Topic relevance is confirmed independently of LDA probability. Daily sentiment is the LDA-probability-weighted mean among relevant, non-missing article-topic scores. Because 226 articles are spread over 92 days, raw daily series are sparse. The figures below therefore use trailing seven-day pooled windows: all qualifying articles in each window are re-aggregated directly, rather than averaging daily means. Missing windows remain missing and are not connected through fabricated zeros.",
    )]
    story += [figure("figures/policy_sentiment/notebook/rolling_7d_all_policy_attention.png", "Figure 2. Trailing seven-day Guardian policy-topic attention. This is the most continuous descriptive series because every valid LDA article contributes its complete probability vector. The measure captures coverage allocation, not sentiment.", width=6.65*inch)]
    story += [PageBreak()]
    story += [P("4. Policy Attention and Sentiment over Time (continued)", "H1x")]
    story += [figure("figures/policy_sentiment/notebook/rolling_7d_all_policy_sentiment.png", "Figure 3. Trailing seven-day topic sentiment presented in Guardian policy texts. Pooling produces a substantially more continuous display than the daily chart while preserving genuine missing windows, especially for the lower-coverage T0 and T1 series. Orange markers identify low effective sample size.", width=6.65*inch)]
    coverage_rows = [
        ["T0", "AI / digital governance", 34, "15.0%", "44.9%", 13, 16],
        ["T1", "Infrastructure / federal investment", 34, "15.0%", "55.6%", 16, 20],
        ["T2", "Trade / health / legal rights", 159, "70.4%", "92.6%", 68, 107],
        ["T3", "Iran / oil / energy security", 114, "50.4%", "91.5%", 23, 57],
        ["T4", "Immigration / household economy", 122, "54.0%", "83.6%", 42, 68],
        ["T5", "Defense / military security", 93, "41.2%", "93.1%", 31, 52],
    ]
    story += [P("Table 4. Annotation coverage by topic", "H2x"), academic_table(
        ["Topic", "Short name", "Sentiment n", "Article cov.", "Weight cov.", "Dem. n", "Rep. n"],
        coverage_rows, [0.42*inch, 2.28*inch, 0.72*inch, 0.72*inch, 0.72*inch, 0.55*inch, 0.55*inch], 6.3,
        right_cols={2,3,4,5,6}
    )]
    story += [P(
        "The annotation stage processed all 226 articles (1,356 article-topic records), with no failed articles and no evidence strings that could not be located in the source text. Nevertheless, the label distribution is highly asymmetric: topic sentiment is predominantly negative, Democratic opposition is absent, and Republican opposition dominates the non-neutral Republican labels. This may reflect the sample period, Guardian framing, and model behavior; it is not a validated measure of electoral opinion.", "Smallx")]
    story += [PageBreak()]

    # 5 sweep
    story += [P("5. Relationship with the Democratic Sweep Market", "H1x")]
    story += [P(
        "The market series is the Polymarket Yes price for Democrats controlling both chambers after the 2026 midterms. Hourly observations are converted from UTC to America/New_York, validated to lie in [0,1], filtered to each requested segment, sorted, and deduplicated. Fourteen API records appended outside their requested historical segments were removed. All 92 daily closing proxies pass the three-hour staleness rule. Prices are market-implied probability proxies, not objective probabilities.",
    )]
    story += [figure("figures/sweep_analysis/democratic_sweep_daily_probability.png", "Figure 4. Democratic Sweep daily closing price proxy, July 10-October 9, 2026. The contract is the joint Senate-and-House event; separate chamber probabilities are neither substituted nor multiplied.", width=6.55*inch)]
    main = sweep_corr[
        (sweep_corr.price_measure == "p_close")
        & (sweep_corr.relation == "change_same")
        & (sweep_corr.method == "pearson")
        & (sweep_corr.index_type.isin(["topic_sentiment", "democrat_stance", "republican_stance"]))
    ].copy()
    idx_order = {"topic_sentiment": 0, "democrat_stance": 1, "republican_stance": 2}
    main["_o"] = main.index_type.map(idx_order)
    main = main.sort_values(["_o", "topic_id"])
    corr_rows = []
    for _, r in main.iterrows():
        label = {"topic_sentiment":"Topic sentiment", "democrat_stance":"Democratic stance", "republican_stance":"Republican stance"}[r.index_type]
        ci = "--" if pd.isna(r.bootstrap_ci_low) else f"[{r.bootstrap_ci_low:.3f}, {r.bootstrap_ci_high:.3f}]"
        corr_rows.append([label, f"T{int(r.topic_id)}", fmt_num(r.coefficient,3), int(r.effective_n), ci, fmt_q(r.bh_q), "Yes" if safe_bool(r.bh_reject_0_05) else "No"])
    story += [P("Table 5. Primary same-day change tests: corr(dS<t>, dP<t>)", "H2x"), academic_table(
        ["Index", "Topic", "Pearson r", "N", "7-day block 95% CI", "BH q", "q<.05"],
        corr_rows, [1.25*inch,0.42*inch,0.67*inch,0.38*inch,1.28*inch,0.62*inch,0.52*inch], 6.2,
        right_cols={2,3,5}
    )]
    story += [P(
        "No primary correlation survives the 36-test Benjamini-Hochberg correction. The largest positive coefficients occur for Republican stance in T4 (r=0.313), T5 (r=0.312), T2 (r=0.306), and T3 (r=0.302), but every adjusted q-value is 0.587 or larger and every block-bootstrap interval crosses zero. T2 topic sentiment has r=0.170 with N=61; its positive bootstrap interval does not translate into multiplicity-adjusted significance (q=0.587). The Democratic Sweep results therefore do not establish a statistically significant relationship during the study window.",
    )]
    story += [PageBreak()]

    # 6 sweep robustness and OOS
    story += [P("6. Democratic Sweep Robustness and Prediction", "H1x")]
    story += [figure("figures/sweep_analysis/same_day_change_correlation_summary.png", "Figure 5. Primary same-day change correlations. Confidence intervals use a seven-day moving-block bootstrap that preserves calendar order and missingness. None of the estimates survives BH adjustment.", width=6.55*inch)]
    story += [P(
        "The expanding-window backtest uses S<t> observed at the end of day t to predict the next-day market change dP<t+1>. Estimation begins only after 30 historical feature-target pairs, missing feature days are skipped, and standardization parameters are estimated from the then-available training window. Two benchmarks are evaluated on exactly the same forecast dates: zero change and the historical training-window mean change.",
    )]
    winners = sweep_oos[
        (sweep_oos.status == "completed")
        & sweep_oos.beats_zero_rmse.map(safe_bool)
        & sweep_oos.beats_train_mean_rmse.map(safe_bool)
    ].copy()
    oos_rows = []
    for _, r in winners.sort_values("oos_r2_vs_zero", ascending=False).iterrows():
        label = {"topic_sentiment":"Topic sentiment", "democrat_stance":"Democratic stance", "republican_stance":"Republican stance"}[r.index_type]
        oos_rows.append([
            label, f"T{int(r.topic_id)}", r.feature_variant, int(r.prediction_count),
            fmt_num(r.model_rmse,5), fmt_num(r.zero_rmse,5), fmt_num(r.train_mean_rmse,5),
            f"{r.oos_r2_vs_zero:.3f}", f"{r.oos_r2_vs_train_mean:.3f}",
        ])
    story += [P("Table 6. Walk-forward models that beat both RMSE benchmarks", "H2x"), academic_table(
        ["Index", "Topic", "Feature", "Pred. N", "Model", "Zero", "Mean", "R2 vs 0", "R2 vs mean"],
        oos_rows, [1.18*inch,0.4*inch,0.48*inch,0.52*inch,0.62*inch,0.62*inch,0.62*inch,0.65*inch,0.72*inch], 6.15,
        right_cols={3,4,5,6,7,8}
    )]
    story += [callout(
        "Prediction result",
        "Four of 15 evaluable specifications beat both RMSE benchmarks. The best is T2 Democratic stance in levels (18 forecasts; OOS R2=0.091 versus zero and 0.028 versus the training mean). The results indicate modest predictive content in selected policy-news signals during the study window.",
    )]
    story += [PageBreak()]

    # 7 stocks
    story += [P("7. Individual-Stock Associations", "H1x")]
    story += [P(
        "For each stock and each sufficiently observed topic index, the project estimates a univariate HAC regression of market-adjusted open-to-close return on the pre-open policy signal. Effects are reported in basis points for a one-standard-deviation increase in the signal. The main family contains 9,976 valid tests after enforcing a fixed minimum of 30 observations. Benjamini-Hochberg correction is applied globally across this family; a stricter subset additionally requires a five-day moving-block bootstrap interval to exclude zero in the HAC direction.",
    )]
    count_rows = []
    grouped = stock_fdr.groupby(["topic_id", "index_type"]).size().sort_values(ascending=False)
    strict_grouped = stock_strict.groupby(["topic_id", "index_type"]).size()
    for (tid, idx), n in grouped.items():
        label = {"topic_sentiment":"Topic sentiment", "democrat_stance":"Democratic stance", "republican_stance":"Republican stance"}[idx]
        count_rows.append([f"T{int(tid)}", label, int(n), int(strict_grouped.get((tid, idx), 0))])
    story += [P("Table 7. Significant stock-index tests by signal", "H2x"), academic_table(
        ["Topic", "Index", "BH-FDR q<.05", "+ block bootstrap"], count_rows,
        [0.6*inch, 2.3*inch, 1.25*inch, 1.35*inch], 7.0, right_cols={2,3}
    )]
    story += [P(
        "Across 998 testable stocks, 346 stock-index rows (299 unique stocks) pass global FDR, and 161 rows (148 unique stocks) also pass the block-bootstrap screen. Discoveries are concentrated in T4 Democratic stance (147 FDR rows) and T2 Democratic stance (120), followed by T3 topic sentiment (24) and T5 topic sentiment (19). Concentration in a few sparse news signals increases the importance of robustness and economic interpretation.",
    )]
    reps = stock_strict.sort_values(["q_value", "raw_p_value"]).head(12)
    rep_rows = []
    for _, r in reps.iterrows():
        label = {"topic_sentiment":"Sentiment", "democrat_stance":"Dem. stance", "republican_stance":"Rep. stance"}[r.index_type]
        rep_rows.append([r.ticker, f"T{int(r.topic_id)}", label, int(r.N), f"{r.standardized_effect_bps:+.1f}", fmt_q(r.q_value), "Yes"])
    story += [P("Table 8. Representative strict stock findings", "H2x"), academic_table(
        ["Ticker", "Topic", "Index", "N", "bps / 1SD", "BH q", "Bootstrap"], rep_rows,
        [0.65*inch,0.45*inch,1.15*inch,0.42*inch,0.8*inch,0.78*inch,0.65*inch], 6.6,
        right_cols={3,4,5}
    )]
    story += [PageBreak()]

    # 8 stock figure and caveat
    story += [P("8. Equity Findings: Magnitude and Interpretation", "H1x")]
    story += [stock_effect_chart(reps)]
    story += [P(
        "Representative strict findings include negative T4 Democratic-stance associations for Essential Properties Realty Trust (ESS), Cheniere Energy (LNG), Texas Instruments (TXN), and Chipotle (CMG), alongside positive associations for Macy's (M), The Toro Company (TTC), and Pool Corporation (POOL). T2 Democratic stance is negatively associated with Rockwell Automation (ROK) and Caterpillar (CAT) but positively associated with PG&E (PCG). These signs describe within-sample co-movement with a text-derived signal; they do not imply that a party stance caused the return.",
    )]
    story += [callout(
        "Statistical significance",
        "Global BH-FDR controls the expected false-discovery proportion across the tested stock-index family, while the moving-block bootstrap evaluates stability under time dependence. Together, 346 rows pass FDR and 161 also pass the bootstrap screen, showing that a substantial subset of the equity relationships is statistically robust.",
    )]
    story += [PageBreak()]

    # 9 basket long-only
    story += [P("9. Dedicated Midterm Baskets: Long-Only Results", "H1x")]
    story += [P(
        "Dedicated baskets pool stocks associated with the same topic-index signal. Long-only baskets equally weight constituent returns. FDR baskets use stocks with globally adjusted q-values below 0.05, while the stricter bootstrap baskets retain only relationships whose moving-block confidence intervals exclude zero in the same direction. This construction summarizes the common market response across stocks linked to each policy signal.",
    )]
    sig = baskets[(baskets.scope == "signal_specific") & (baskets.fdr_significant == True)].copy()
    long_sig = sig[sig.direction_mode == "long_only_equal_weight"].sort_values("q_value")
    long_rows = []
    for _, r in long_sig.iterrows():
        rule = "Strict signal" if str(r.basket_id).startswith("BOOTSTRAP_SIGNAL") else ("Strict top 20" if str(r.basket_id).startswith("BOOTSTRAP_TOP20") else "FDR signal")
        idx = {"topic_sentiment":"Sentiment", "democrat_stance":"Dem. stance", "republican_stance":"Rep. stance"}[r.index_type]
        long_rows.append([rule, f"T{int(r.topic_id)} {idx}", int(r.component_count), int(r.N), f"{r.standardized_effect_bps:+.1f}", fmt_q(r.q_value), "Yes" if safe_bool(r.fdr_and_bootstrap_supported) else "No"])
    story += [P("Table 9. All FDR-significant dedicated long-only baskets", "H2x"), academic_table(
        ["Selection", "Signal", "Stocks", "N", "bps / 1SD", "BH q", "Basket bootstrap"],
        long_rows, [1.15*inch,1.3*inch,0.52*inch,0.42*inch,0.8*inch,0.72*inch,0.9*inch], 6.15,
        right_cols={2,3,4,5}
    )]
    story += [P(
        "There are 13 significant dedicated long-only baskets. The strongest are the T2 Democratic-stance strict baskets: 21-stock and top-20 versions return approximately -29 basis points per one-standard-deviation increase in the index (q values below 1e-11). T3 Republican stance and T3 sentiment long-only baskets are positive, while T4 sentiment baskets are negative. These collective patterns indicate economically meaningful topic-specific relationships in market-adjusted stock returns.",
    )]
    story += [PageBreak()]

    # 10 basket directional
    story += [P("10. Dedicated Midterm Baskets: Direction-Aligned Results", "H1x")]
    story += [P(
        "Direction-aligned baskets multiply each constituent by the sign of its stock-level coefficient before equal weighting. This creates a long-short statistical factor that measures whether individual-stock responses aggregate coherently at the topic level. Reported effects are gross statistical sensitivities and do not incorporate borrow costs, turnover, spreads, or capacity.",
    )]
    dir_sig = sig[sig.direction_mode == "direction_aligned_equal_weight"].sort_values("q_value")
    dir_rows = []
    for _, r in dir_sig.iterrows():
        rule = "Strict signal" if str(r.basket_id).startswith("BOOTSTRAP_SIGNAL") else "Strict top 20"
        idx = {"topic_sentiment":"Sentiment", "democrat_stance":"Dem. stance", "republican_stance":"Rep. stance"}[r.index_type]
        dir_rows.append([rule, f"T{int(r.topic_id)} {idx}", int(r.component_count), int(r.N), f"{r.standardized_effect_bps:+.1f}", fmt_q(r.q_value), "Yes" if safe_bool(r.fdr_and_bootstrap_supported) else "No"])
    story += [P("Table 10. All FDR-significant dedicated direction-aligned baskets", "H2x"), academic_table(
        ["Selection", "Signal", "Stocks", "N", "bps / 1SD", "BH q", "Basket bootstrap"],
        dir_rows, [1.15*inch,1.3*inch,0.52*inch,0.42*inch,0.8*inch,0.72*inch,0.9*inch], 6.15,
        right_cols={2,3,4,5}
    )]
    story += [P(
        "All 13 direction-aligned dedicated baskets pass both basket-level FDR and block-bootstrap criteria. Estimated magnitudes range from +45 to +104 basis points per signal standard deviation. The results show strong cross-stock coherence and economically meaningful aggregate sensitivity to the corresponding policy-news indices.",
    )]
    story += [PageBreak()]

    # 11 conclusion
    story += [P("11. Conclusions, Limitations, and Reproducibility", "H1x")]
    conclusion_rows = [
        ["Topic modeling", "Useful descriptive structure", "K=6 has the best candidate coherence (0.407), low keyword overlap, and interpretable policy clusters."],
        ["Time series", "Use pooled 7-day displays", "They are more continuous than raw daily sentiment while retaining genuine missingness and sample-size warnings."],
        ["Democratic Sweep", "No reliable association", "No primary dS-dP correlation survives BH correction; OOS gains are small and sparse."],
        ["Individual stocks", "Many significant associations", "346 FDR rows / 299 stocks; 161 rows / 148 stocks also pass block bootstrap."],
        ["Dedicated baskets", "Strong statistical associations", "26 signal-specific baskets are FDR-significant; 24 also pass the basket-level block-bootstrap criterion."],
    ]
    story += [P("Table 11. Evidence summary", "H2x"), academic_table(
        ["Component", "Bottom line", "Evidence"], conclusion_rows,
        [1.15*inch,1.55*inch,3.95*inch], 7.0
    )]
    story += [P("11.1 Main limitations", "H2x")]
    story += [P(
        "First, the news sample is produced by a single publisher and cannot represent the electorate. Second, policy selection and topic labels involve judgment; T2 and T4 remain broad composites. Third, sentiment labels lack an independent human gold standard, and party-stance labels are asymmetric. Fourth, daily sentiment is sparse, especially for T0 and T1. Fifth, Polymarket liquidity, microstructure, and price formation are not modeled. Sixth, equity tests are observational, so the estimates describe statistical relationships rather than causal effects.",
    )]
    story += [P("11.2 Reproducibility checks", "H2x")]
    story += [P(
        "Saved validation reports confirm: article IDs align across raw text, LDA probabilities, and annotations; all 1,356 article-topic keys are unique; topic probabilities sum to one within numerical tolerance; saved LDA, dictionary, and phrase models reload successfully; daily calendars contain all 92 dates; market timestamps are unique and in range; differencing occurs before complete-case filtering; walk-forward targets are causally available; and manual recomputations match one daily sentiment value, one weekly value, one correlation, and one forecast.",
    )]
    story += [P("Selected project sources", "H2x")]
    story += [P(
        'The Guardian Open Platform (news corpus); spaCy and Gensim (text preprocessing and LDA); DeepSeek API (structured article annotation); Polymarket Gamma and CLOB APIs (contract verification and hourly price history); yfinance (equity and SPY price data). All numerical results in this report are read from the final validated project outputs under <i>data/lda_policy</i>, <i>data/policy_sentiment</i>, <i>data/sweep_analysis</i>, and <i>data/equity_sentiment</i>.<br/><b>Project repository:</b> <link href="https://github.com/kc6534-JerryCai/NLP_asignment5"><font color="#294A6D"><u>https://github.com/kc6534-JerryCai/NLP_asignment5</u></font></link>', "Smallx")]
    return story


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = ReportDoc(str(OUT))
    story = build_story()
    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    main()
