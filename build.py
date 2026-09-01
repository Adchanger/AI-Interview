#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-Interview 静态站点生成器

扫描 docs/ 下的 markdown 知识文档，生成纯静态 HTML 站点到 site/：
  - site/index.html      首页（分类浏览 + 搜索过滤 + 最近更新条）
  - site/updates.html    最近更新（近三天 + 日期筛选 + 完整时间线）
  - site/<分类>/<文章>.html  阅读页（目录、双链跳转、上一篇/下一篇）

用法：.venv/bin/python3 build.py
无外部服务依赖，生成后直接双击 site/index.html 即可离线浏览。
"""

import datetime
import html
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown
from latex2mathml.converter import convert as latex_to_mathml
from markdown.extensions import Extension
from markdown.extensions.toc import slugify_unicode
from markdown.preprocessors import Preprocessor
from pygments.formatters import HtmlFormatter

# ------------------------------------------------------------ 配置

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
SRC_DIR = ROOT / "site-src"
OUT_DIR = ROOT / "site"

SITE = {
    "title": "AI Interview 知识库",
    "brand": "AI Interview",
    "tagline": "AI 方向知识点整理与面试准备 · 每天学习一点点，持续更新中",
    "recent_days": 3,          # 「最近 N 天」窗口
    "recent_strip_count": 6,   # 首页「最近更新」横条展示条数
    "date_chip_count": 12,     # 更新页快捷日期 chips 数量
    "tag_cloud_count": 16,     # 首页侧边栏「热门标签」个数
}

# 板块：分类之上的一层，让首页侧边栏呈现「基础理论 → 大模型技术 → 工程与面试」的层级
GROUPS = {
    "foundation": {"name": "基础理论",   "icon": "🌱", "order": 10,
                   "desc": "机器学习与深度学习的底层基本功"},
    "frontier":   {"name": "大模型技术", "icon": "🚀", "order": 20,
                   "desc": "LLM 原理、检索增强、Agent 与多模态"},
    "practice":   {"name": "工程与面试", "icon": "🛠️", "order": 30,
                   "desc": "训练推理落地实践与面试准备"},
}
DEFAULT_GROUP = {"name": "其他", "icon": "📚", "order": 999, "desc": ""}

# 分类（docs/ 下一级目录）展示名、图标与所属板块；未登记的目录按目录名显示
CATEGORIES = {
    "ml":          {"name": "机器学习",   "icon": "📊", "order": 10, "group": "foundation"},
    "dl":          {"name": "深度学习",   "icon": "🧠", "order": 20, "group": "foundation"},
    "nlp":         {"name": "自然语言处理", "icon": "💬", "order": 45, "group": "foundation"},
    "llm":         {"name": "大语言模型", "icon": "🤖", "order": 30, "group": "frontier"},
    "rag":         {"name": "RAG 检索",  "icon": "🔎", "order": 35, "group": "frontier"},
    "agent":       {"name": "Agent",     "icon": "🦾", "order": 38, "group": "frontier"},
    "multimodal":  {"name": "多模态",     "icon": "🎨", "order": 40, "group": "frontier"},
    "engineering": {"name": "工程实践",   "icon": "🛠️", "order": 50, "group": "practice"},
    "interview":   {"name": "面试题",     "icon": "💼", "order": 60, "group": "practice"},
}
DEFAULT_CATEGORY = {"name": None, "icon": "📚", "order": 999, "group": None}

WEEKDAYS = "一二三四五六日"

# ------------------------------------------------------------ 数据模型

class Doc:
    def __init__(self, md_path: Path):
        self.md_path = md_path
        self.rel_md = md_path.relative_to(ROOT).as_posix()      # docs/llm/xxx.md
        self.category = md_path.parent.name                      # llm
        self.slug = md_path.stem                                 # xxx
        # 站点内输出路径（镜像 docs 结构，图片相对路径 ../images 自动生效）
        self.out_rel = Path(self.rel_md).relative_to("docs").with_suffix(".html")  # llm/xxx.html
        self.out_path = OUT_DIR / self.out_rel

        raw = md_path.read_text(encoding="utf-8")
        self.title = self._parse_title(raw) or self.slug
        self.date = self._parse_date(raw) or self._git_date() or \
            datetime.date.fromtimestamp(md_path.stat().st_mtime).isoformat()
        self.tags = self._parse_tags(raw)
        self.excerpt = self._parse_excerpt(raw)
        self.body_md = self._strip_meta(raw)

    def _parse_title(self, raw):
        m = re.search(r"^#\s+(.+?)\s*$", raw, re.M)
        return m.group(1) if m else None

    def _parse_date(self, raw):
        m = re.search(r"^>\s*\*\*更新时间\*\*[:：]\s*(\d{4}-\d{2}-\d{2})", raw, re.M)
        return m.group(1) if m else None

    def _git_date(self):
        try:
            out = subprocess.run(
                ["git", "log", "-1", "--format=%cs", "--", self.rel_md],
                capture_output=True, text=True, cwd=ROOT, timeout=10,
            ).stdout.strip()
            return out or None
        except Exception:
            return None

    def _parse_tags(self, raw):
        m = re.search(r"^>\s*\*\*标签\*\*[:：]\s*(.+?)\s*$", raw, re.M)
        if not m:
            return []
        # 只按中英文标点切分；不切空格，否则「KV Cache」这类带空格的标签会被拆成两个
        parts = re.split(r"[、，,;/；]+", m.group(1))
        return [p.strip() for p in parts if p.strip()]

    def _parse_excerpt(self, raw):
        m = re.search(r"^>\s*\*\*一句话\*\*[:：]\s*(.+?)\s*$", raw, re.M)
        if not m:
            return ""
        text = m.group(1)
        text = re.sub(r"\[\[([^\[\]]+?\.md)\]\]", lambda mm: Path(mm.group(1)).stem, text)
        text = re.sub(r"[*`#>\[\]]", "", text)
        return text.strip()

    def _strip_meta(self, raw):
        """移除正文中的 H1 与「更新时间 / 标签」元信息行（页面头部会统一展示）。"""
        lines = raw.splitlines()
        out, h1_removed = [], False
        for line in lines:
            if not h1_removed and re.match(r"^#\s+", line):
                h1_removed = True
                continue
            if re.match(r"^>\s*\*\*(更新时间|标签)\*\*[:：]", line):
                continue
            out.append(line)
        return "\n".join(out)

    @property
    def category_name(self):
        return CATEGORIES.get(self.category, DEFAULT_CATEGORY)["name"] or self.category

    @property
    def category_icon(self):
        return CATEGORIES.get(self.category, DEFAULT_CATEGORY)["icon"]


# ------------------------------------------------------------ 工具

def esc(s):
    return html.escape(s, quote=True)


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def rel_href(from_out_rel: Path, to_out_rel) -> str:
    """从某个输出页面到另一个输出文件（或目录相对路径）的相对链接。"""
    return os.path.relpath(str(to_out_rel), from_out_rel.parent.as_posix()).replace(os.sep, "/")


def detect_repo_url():
    env = os.environ.get("GITHUB_REPOSITORY")
    if env:
        return f"https://github.com/{env}"
    try:
        url = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True, text=True, cwd=ROOT, timeout=10,
        ).stdout.strip()
    except Exception:
        return ""
    if not url:
        return ""
    m = re.match(r"git@([^:]+):(.+?)(?:\.git)?$", url)
    if m:
        return f"https://{m.group(1)}/{m.group(2)}"
    return re.sub(r"\.git$", "", url)


MATH_SCAN = re.compile(
    r"""
      (?P<fence>`+)(?P<fencebody>.+?)(?P=fence)      # 行内代码，原样保留（先匹配以保护其中的 $）
    | \$\$(?P<block>(?:[^$\\]|\\.)+?)\$\$            # 独立展示公式
    | (?<![\\$0-9A-Za-z])
      \$(?!\s)(?P<inline>(?:[^$\n\\]|\\.)+?)(?<!\s)\$
      (?![0-9$])                                     # 行内公式（排除「$5」这类货币写法）
    """,
    re.S | re.X,
)


class MathPreprocessor(Preprocessor):
    """把 $...$ / $$...$$ 转成 MathML 并塞进 htmlStash，避开代码与后续 markdown 处理。

    优先级低于 fenced_code_block（25），因此 ``` 围栏里的内容早已被暂存，不会被误转；
    行内 `code` 由 MATH_SCAN 的第一个分支保护。
    """

    priority = 15

    def run(self, lines):
        def sub(m):
            if m.group("fence") is not None:
                return m.group(0)                    # 行内代码：原样返回
            block = m.group("block")
            tex = (block if block is not None else m.group("inline")).strip()
            try:
                mathml = latex_to_mathml(
                    tex, display="block" if block is not None else "inline")
            except Exception as exc:                 # 公式写错时不要让整站构建失败
                print(f"   ⚠️  公式转换失败：{tex[:60]!r} → {exc}")
                return m.group(0)
            cls = "math-block" if block is not None else "math-inline"
            html_frag = f'<span class="{cls}">{mathml}</span>'
            return self.md.htmlStash.store(html_frag)

        return MATH_SCAN.sub(sub, "\n".join(lines)).split("\n")


class MathExtension(Extension):
    def extendMarkdown(self, md):
        md.preprocessors.register(MathPreprocessor(md), "mathml", MathPreprocessor.priority)


MD = markdown.Markdown(
    extensions=["extra", "toc", "codehilite", "sane_lists", MathExtension()],
    extension_configs={
        "codehilite": {"css_class": "highlight", "guess_lang": False},
        "toc": {"slugify": slugify_unicode},
    },
)

WIKI_RE = re.compile(r"\[\[([^\[\]]+?\.md)\]\]")


def render_markdown(doc: Doc, docs_by_rel: dict) -> str:
    """转换正文：先把 [[双链]] 替换为 HTML 锚点，再走 markdown。"""
    def sub(m):
        target_rel = m.group(1).lstrip("/")          # docs/llm/xxx.md
        target = docs_by_rel.get(target_rel)
        if target:
            href = rel_href(doc.out_rel, target.out_rel)
            return f'<a class="wikilink" href="{esc(href)}">{esc(target.title)}</a>'
        name = Path(target_rel).stem
        return f'<span class="wikilink missing" title="本文尚未发布">{esc(name)}</span>'

    text = WIKI_RE.sub(sub, doc.body_md)
    MD.reset()
    return MD.convert(text)


def extract_toc(body_html: str):
    """从生成的 HTML 中提取 h2/h3 目录。

    标题里若含 MathML（$...$ 写法），剥标签会把 <msqrt><mi>d</mi><mi>k</mi>
    压成无意义的 "dk"，所以整块公式统一降级成 ⟨公式⟩ 占位，提示改用纯文本写标题。
    """
    toc = []
    for m in re.finditer(r'<h([23]) id="([^"]+)"[^>]*>(.*?)</h\1>', body_html, re.S):
        inner = re.sub(r'<span class="math-[^"]*">.*?</span>\s*', "⟨公式⟩", m.group(3), flags=re.S)
        inner = re.sub(r"<[^>]+>", "", inner)
        toc.append((int(m.group(1)), m.group(2), html.unescape(inner).strip()))
    return toc


# ------------------------------------------------------------ 页面模板

def page_shell(*, page, title, out_rel, nav_html, content, repo_url, build_date, extra_top=""):
    root = rel_href(out_rel, ".")
    root = "" if root == "." else root + "/"
    gh = (f'<a class="nav-github" href="{esc(repo_url)}" target="_blank" rel="noopener">GitHub ↗</a>'
          if repo_url else "")
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{esc(title)} · {esc(SITE["title"])}</title>
<meta name="description" content="{esc(SITE["tagline"])}">
<link rel="stylesheet" href="{root}assets/style.css">
<link rel="stylesheet" href="{root}assets/pygments.css">
</head>
<body data-page="{page}">
{extra_top}
<nav class="site-nav">
  <a class="nav-brand" href="{root}index.html">
    <span class="logo">✦</span><span class="brand-text">{esc(SITE["brand"])}</span>
  </a>
  <div class="nav-links">
    {nav_html}
    {gh}
  </div>
</nav>
<main class="page-main">
{content}
</main>
<footer class="site-footer">
  <p class="foot-line">{esc(SITE["title"])} · 每天学习一点点</p>
  <p class="foot-line">页面由 <code>build.py</code> 自动生成 · 构建于 {build_date}{' · ' + f'<a href="{esc(repo_url)}" target="_blank" rel="noopener">GitHub 仓库</a>' if repo_url else ''}</p>
</footer>
<script src="{root}assets/app.js" defer></script>
</body>
</html>
"""


def nav_links(page, out_rel, repo_url):
    items = [
        ("home", "首页", "index.html"),
        ("updates", "最近更新", "updates.html"),
    ]
    parts = []
    for key, label, target in items:
        href = rel_href(out_rel, target)
        cls = ' class="active"' if key == page else ""
        parts.append(f'<a href="{esc(href)}"{cls}>{label}</a>')
    return "\n    ".join(parts)


def date_badge(date_str, fresh_dates):
    cls = "date-badge fresh" if date_str in fresh_dates else "date-badge"
    return f'<span class="{cls}">🕐 {esc(date_str)}</span>'


def tag_chips(tags):
    return "".join(f'<span class="tag-chip">{esc(t)}</span>' for t in tags[:5])


def article_card(doc: Doc, from_out_rel: Path, fresh_dates, show_category=False):
    href = rel_href(from_out_rel, doc.out_rel)
    cat = f'<span class="tag-chip">{esc(doc.category_icon)} {esc(doc.category_name)}</span>' if show_category else ""
    excerpt = f'<p class="card-excerpt">{esc(doc.excerpt)}</p>' if doc.excerpt else ""
    search = esc(" ".join([doc.title, " ".join(doc.tags), doc.category_name]).lower())
    return f"""<a class="article-card" href="{esc(href)}" data-search="{search}">
  <h3 class="card-title">{esc(doc.title)}</h3>
  {excerpt}
  <div class="card-foot">{date_badge(doc.date, fresh_dates)}{tag_chips(doc.tags)}{cat}</div>
</a>"""


# ------------------------------------------------------------ 各页面生成

def sidebar_tree(groups, out_rel, recent_keys, top_tags, total):
    """首页左侧的层级索引：板块 → 分类 → 文章。"""
    node_all = f"""<a class="tree-node tree-all is-active" href="#" data-cat="">
      <span class="t-icon">🗂</span>
      <span class="t-name">全部分类</span>
      <span class="t-count">{total}</span>
    </a>"""

    blocks = []
    for g in groups:
        cat_nodes = []
        for cat in g["cats"]:
            items = []
            for d in cat["docs"]:
                href = rel_href(out_rel, d.out_rel)
                search = esc(" ".join([d.title, " ".join(d.tags), d.category_name]).lower())
                # 小圆点 = 该篇出现在上方「最近更新」里，避免全站同一天更新时满屏都是点
                dot = '<span class="new-dot" title="最近更新"></span>' if d.rel_md in recent_keys else ""
                items.append(
                    f'<li><a href="{esc(href)}" data-search="{search}">'
                    f'{dot}<span class="sub-title">{esc(d.title)}</span></a></li>')
            cat_nodes.append(f"""<div class="tree-node tree-cat" data-cat="{esc(cat['key'])}">
      <button class="t-row" type="button" aria-expanded="false">
        <span class="t-caret">›</span>
        <span class="t-icon">{esc(cat['icon'])}</span>
        <span class="t-name">{esc(cat['name'])}</span>
        <span class="t-count">{len(cat['docs'])}</span>
      </button>
      <ul class="t-sub">
        {chr(10).join(items)}
      </ul>
    </div>""")
        blocks.append(f"""<section class="tree-group" data-group="{esc(g['key'])}">
    <div class="tree-group-head">
      <span class="g-icon">{esc(g['icon'])}</span>
      <span class="g-name">{esc(g['name'])}</span>
      <span class="g-count">{g['count']}</span>
    </div>
    <div class="tree-group-body">
      {chr(10).join(cat_nodes)}
    </div>
  </section>""")

    tags_html = ""
    if top_tags:
        pills = "\n".join(
            f'      <button class="tag-pill" type="button" data-tag="{esc(t)}">'
            f'{esc(t)}<span class="p-n">{n}</span></button>'
            for t, n in top_tags)
        tags_html = f"""<section class="tree-group tree-tags">
    <div class="tree-group-head">
      <span class="g-icon">🏷</span>
      <span class="g-name">热门标签</span>
    </div>
    <div class="tag-cloud">
{pills}
    </div>
  </section>"""

    return node_all + "\n  " + "\n  ".join(blocks) + ("\n  " + tags_html if tags_html else "")


def build_home(docs, groups, fresh_dates, repo_url, build_date, latest_date, top_tags):
    out_rel = Path("index.html")
    cat_count = sum(len(g["cats"]) for g in groups)
    recent = sorted(docs, key=lambda d: (d.date, d.title), reverse=True)[:SITE["recent_strip_count"]]
    recent_html = "\n".join(
        f"""<a class="recent-item" href="{esc(rel_href(out_rel, d.out_rel))}">
  <span class="r-date">{esc(d.date)}</span>
  <span class="r-title">{esc(d.title)}</span>
  <span class="r-cat">{esc(d.category_name)}</span>
</a>""" for d in recent
    )

    # 主内容区：板块 → 分类 → 文章卡片
    group_blocks = []
    for g in groups:
        cat_blocks = []
        for cat in g["cats"]:
            cards = "\n".join(article_card(d, out_rel, fresh_dates) for d in cat["docs"])
            cat_blocks.append(f"""<section class="category-section" data-cat="{esc(cat['key'])}">
    <div class="category-head">
      <span class="icon">{esc(cat['icon'])}</span>
      <h2>{esc(cat['name'])}</h2>
      <span class="count">{len(cat['docs'])} 篇</span>
      <span class="dir">docs/{esc(cat['key'])}/</span>
    </div>
    <div class="article-grid">
{cards}
    </div>
  </section>""")
        desc = f'<span class="g-desc">{esc(g["desc"])}</span>' if g["desc"] else ""
        group_blocks.append(f"""<section class="cat-group" data-group="{esc(g['key'])}">
  <header class="cat-group-head">
    <span class="g-icon">{esc(g['icon'])}</span>
    <h2>{esc(g['name'])}</h2>
    {desc}
    <span class="g-count">{g['count']} 篇</span>
  </header>
  <div class="cat-group-body">
    {chr(10).join(cat_blocks)}
  </div>
</section>""")

    tree_html = sidebar_tree(groups, out_rel, {d.rel_md for d in recent}, top_tags, len(docs))

    content = f"""
<div class="page-shell">
  <aside class="side-rail" id="side-rail">
    <div class="rail-head">
      <button class="rail-toggle" type="button" aria-expanded="false" aria-controls="site-tree">
        <span class="st-icon">🗂</span>
        <span class="st-text">全站索引</span>
        <span class="st-count">{len(docs)} 篇 · {cat_count} 类</span>
        <span class="st-caret">▾</span>
      </button>
      <div class="rail-search">
        <input type="search" placeholder="搜索文章标题、标签…" aria-label="搜索文章">
      </div>
    </div>
    <nav class="tree" id="site-tree" aria-label="全站分类索引">
  {tree_html}
    </nav>
  </aside>

  <div class="content-col" id="content-col">
    <section class="hero">
      <h1>每天学习一点 <span class="em">AI</span> 知识</h1>
      <p class="tagline">{esc(SITE["tagline"])}</p>
      <div class="hero-stats">
        <div class="stat"><div class="num">{len(docs)}</div><div class="label">已收录文章</div></div>
        <div class="stat"><div class="num">{cat_count}</div><div class="label">知识分类</div></div>
        <div class="stat"><div class="num">{esc(latest_date[5:])}</div><div class="label">最近更新</div></div>
      </div>
    </section>

    <section class="recent-strip">
      <h2>🌿 最近更新 <a class="more" href="updates.html">查看全部 →</a></h2>
      <div class="recent-list">
{recent_html}
      </div>
    </section>

    <div class="view-bar">
      <span class="vb-title">🗂 全部分类</span>
      <span class="vb-desc">共 {len(docs)} 篇文章 · 按板块与分类浏览</span>
      <button class="vb-all" type="button" hidden>← 返回全部分类</button>
    </div>
    <div class="filter-empty" style="display:none">没有找到匹配的文章，换个关键词试试～</div>

{chr(10).join(group_blocks)}
  </div>
</div>
"""
    return page_shell(page="home", title="首页", out_rel=out_rel,
                      nav_html=nav_links("home", out_rel, repo_url),
                      content=content, repo_url=repo_url, build_date=build_date)


def build_updates(docs, fresh_dates, repo_url, build_date, today):
    out_rel = Path("updates.html")
    by_date = {}
    for d in docs:
        by_date.setdefault(d.date, []).append(d)
    dates = sorted(by_date.keys(), reverse=True)
    all_dates_set = set(dates)

    recent_docs = sorted([d for d in docs if d.date in fresh_dates],
                         key=lambda d: (d.date, d.title), reverse=True)
    if recent_docs:
        recent_cards = "\n".join(article_card(d, out_rel, fresh_dates, show_category=True)
                                 for d in recent_docs)
        recent_section = f"""<h2 class="section-title">🌱 最近{SITE["recent_days"]}天更新 <span class="hint">{len(recent_docs)} 篇新内容</span></h2>
<div class="article-grid">
{recent_cards}
</div>"""
    else:
        recent_section = f"""<h2 class="section-title">🌱 最近{SITE["recent_days"]}天更新</h2>
<div class="empty-hint">最近{SITE["recent_days"]}天暂无更新，去下面的时间线回顾一下吧～</div>"""

    chips = "\n".join(
        f'<button class="date-chip" data-date="{esc(dt)}">{esc(dt[5:].replace("-", "/"))}</button>'
        for dt in dates[:SITE["date_chip_count"]]
    )

    timeline = []
    for dt in dates:
        day_docs = sorted(by_date[dt], key=lambda d: natural_key(d.title))
        wd = WEEKDAYS[datetime.date.fromisoformat(dt).weekday()]
        is_today = dt == today
        fresh = '<span class="d-fresh">NEW</span>' if dt in fresh_dates else ""
        cards = "\n".join(
            f"""<a class="article-card" href="{esc(rel_href(out_rel, d.out_rel))}">
  <h3 class="card-title">{esc(d.title)}</h3>
  {'<p class="card-excerpt">' + esc(d.excerpt) + '</p>' if d.excerpt else ''}
  <div class="card-foot"><span class="tag-chip">{esc(d.category_icon)} {esc(d.category_name)}</span>{tag_chips(d.tags)}</div>
</a>""" for d in day_docs)
        timeline.append(f"""<section class="day-group{' today' if is_today else ''}" data-date="{esc(dt)}">
  <div class="day-head">
    <span class="d-date">{esc(dt)}</span>
    <span class="d-week">星期{wd}</span>
    <span class="d-count">{len(day_docs)} 篇</span>
    {fresh}
  </div>
  <div class="day-articles">
{cards}
  </div>
</section>""")

    min_date, max_date = dates[-1], dates[0]
    content = f"""
<div class="wrap">
  <section class="updates-hero">
    <h1>🍃 最近更新</h1>
    <p>每天学习一点点 —— 追更从这里开始</p>
  </section>

  <div class="date-picker-card">
    <label for="date-picker">📅 按日期查看：</label>
    <input type="date" id="date-picker" min="{esc(min_date)}" max="{esc(max_date)}">
    <button class="btn-clear">显示全部</button>
  </div>
  <div class="date-chip-row">
{chips}
  </div>

  {recent_section}

  <h2 class="section-title">🗂 全部更新 <span class="hint">按日期分组，共 {len(docs)} 篇</span></h2>
  <div class="filter-empty" style="display:none">这一天没有更新记录，换个日期试试～</div>
  <div class="timeline">
{chr(10).join(timeline)}
  </div>
</div>
"""
    return page_shell(page="updates", title="最近更新", out_rel=out_rel,
                      nav_html=nav_links("updates", out_rel, repo_url),
                      content=content, repo_url=repo_url, build_date=build_date)


def build_article(doc, docs_by_rel, cat_docs, fresh_dates, repo_url, build_date):
    out_rel = doc.out_rel
    root = rel_href(out_rel, ".") + "/"
    body_html = render_markdown(doc, docs_by_rel)
    toc = extract_toc(body_html)

    idx = next(i for i, d in enumerate(cat_docs) if d.slug == doc.slug)
    prev_doc = cat_docs[idx - 1] if idx > 0 else None
    next_doc = cat_docs[idx + 1] if idx < len(cat_docs) - 1 else None

    def pager_link(d, label, cls):
        if not d:
            return f'<a class="{cls} placeholder"><span class="p-label">&nbsp;</span></a>'
        return f"""<a class="{cls}" href="{esc(rel_href(out_rel, d.out_rel))}">
  <div class="p-label">{label}</div>
  <div class="p-title">{esc(d.title)}</div>
</a>"""

    toc_html = ""
    if len(toc) >= 2:
        items = "\n".join(
            f'<li class="toc-h{lv}"><a href="#{esc(anchor)}">{esc(text)}</a></li>'
            for lv, anchor, text in toc
        )
        toc_html = f"""<aside class="toc-sidebar">
  <div class="toc-title">📑 本文目录</div>
  <ul class="toc-list">
{items}
  </ul>
</aside>"""

    meta_tags = tag_chips(doc.tags)
    content = f"""
<div class="reader-layout">
  <article>
    <header class="article-header">
      <div class="breadcrumb">
        <a href="{root}index.html">首页</a><span class="sep">›</span>
        <span>{esc(doc.category_icon)} {esc(doc.category_name)}</span><span class="sep">›</span>
        <span>正文</span>
      </div>
      <h1>{esc(doc.title)}</h1>
      <div class="article-meta">
        {date_badge(doc.date, fresh_dates)}
        {meta_tags}
      </div>
    </header>
    <div class="article-body">
{body_html}
    </div>
    <nav class="pager">
      {pager_link(prev_doc, "← 上一篇", "prev")}
      {pager_link(next_doc, "下一篇 →", "next")}
    </nav>
  </article>
  {toc_html}
</div>
"""
    return page_shell(page="article", title=doc.title, out_rel=out_rel,
                      nav_html=nav_links("article", out_rel, repo_url),
                      content=content, repo_url=repo_url, build_date=build_date,
                      extra_top='<div class="progress-bar"></div>',
                      ) + ""


# ------------------------------------------------------------ 主流程

def main():
    if not DOCS_DIR.is_dir():
        sys.exit(f"未找到文档目录：{DOCS_DIR}")

    md_files = sorted(DOCS_DIR.rglob("*.md"))
    if not md_files:
        sys.exit("docs/ 下没有任何 markdown 文件")

    docs = [Doc(p) for p in md_files]
    docs_by_rel = {d.rel_md: d for d in docs}

    today = datetime.date.today().isoformat()
    window = { (datetime.date.today() - datetime.timedelta(days=i)).isoformat()
               for i in range(SITE["recent_days"]) }
    fresh_dates = window & {d.date for d in docs}      # 有文章的、且在近 N 天内的日期
    latest_date = max(d.date for d in docs)
    build_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    repo_url = detect_repo_url()

    # 分类分组（分类内按标题自然序，保持「八股 01、02…」连载顺序）
    cat_map = {}
    for d in docs:
        cat_map.setdefault(d.category, []).append(d)
    cats = []
    for key, cat_docs in cat_map.items():
        meta = CATEGORIES.get(key, DEFAULT_CATEGORY)
        cat_docs.sort(key=lambda d: natural_key(d.title))
        cats.append({"key": key, "name": meta["name"] or key,
                     "icon": meta["icon"], "docs": cat_docs})
    cats.sort(key=lambda c: (CATEGORIES.get(c["key"], DEFAULT_CATEGORY)["order"], c["key"]))

    # 在分类之上再聚合一层「板块」，供首页侧边栏呈现层级
    grouped = {}
    for c in cats:
        gk = CATEGORIES.get(c["key"], DEFAULT_CATEGORY).get("group")
        grouped.setdefault(gk, []).append(c)
    groups = []
    for gk, g_cats in grouped.items():
        meta = GROUPS.get(gk, DEFAULT_GROUP)
        groups.append({
            "key": gk or "other",
            "name": meta["name"],
            "icon": meta["icon"],
            "desc": meta["desc"],
            "cats": g_cats,
            "count": sum(len(c["docs"]) for c in g_cats),
        })
    groups.sort(key=lambda g: (GROUPS.get(g["key"], DEFAULT_GROUP)["order"], g["key"]))

    # 热门标签（排除「面试八股」这类几乎篇篇都有、没有区分度的通用标签）
    counter = {}
    for d in docs:
        for t in d.tags:
            counter[t] = counter.get(t, 0) + 1
    too_common = max(2, int(len(docs) * 0.6))
    top_tags = sorted(((t, n) for t, n in counter.items() if 1 < n <= too_common),
                      key=lambda kv: (-kv[1], kv[0]))[:SITE["tag_cloud_count"]]

    # 重建输出目录
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    (OUT_DIR / "assets").mkdir(parents=True)

    # 拷贝静态资源与 docs 内的非 md 文件（图片等，保持相对路径有效）
    for name in ("style.css", "app.js"):
        shutil.copy2(SRC_DIR / name, OUT_DIR / "assets" / name)
    for p in DOCS_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() != ".md":
            dest = OUT_DIR / p.relative_to(DOCS_DIR)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)

    # Pygments 高亮样式
    (OUT_DIR / "assets" / "pygments.css").write_text(
        HtmlFormatter(style="friendly").get_style_defs(".highlight"), encoding="utf-8")
    (OUT_DIR / ".nojekyll").write_text("", encoding="utf-8")

    # 生成页面
    (OUT_DIR / "index.html").write_text(
        build_home(docs, groups, fresh_dates, repo_url, build_date, latest_date, top_tags),
        encoding="utf-8")
    (OUT_DIR / "updates.html").write_text(
        build_updates(docs, fresh_dates, repo_url, build_date, today), encoding="utf-8")
    for d in docs:
        cat_docs = cat_map[d.category]
        d.out_path.parent.mkdir(parents=True, exist_ok=True)
        d.out_path.write_text(
            build_article(d, docs_by_rel, cat_docs, fresh_dates, repo_url, build_date),
            encoding="utf-8")

    print(f"✅ 构建完成：{len(docs)} 篇文章，{len(cats)} 个分类")
    print(f"   输出目录：{OUT_DIR}")
    print(f"   本地预览：open {(OUT_DIR / 'index.html')}")
    if repo_url:
        print(f"   仓库地址：{repo_url}")


if __name__ == "__main__":
    main()
