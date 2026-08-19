# AI-Interview

AI 方向知识点整理与面试准备仓库。

## 内容规划

涵盖 AI 方向的常见知识点，包括但不限于：

- 机器学习基础（经典算法、模型评估、特征工程）
- 深度学习（CNN / RNN / Transformer、优化器、正则化）
- 大语言模型（LLM 原理、Prompt Engineering、RAG、Agent、微调）
- 多模态（视觉、语音、图文理解）
- 工程实践（训练与推理优化、部署、评测）
- 经典面试题与手撕代码

> 内容逐步补充中，欢迎补充与交流。

## 结构

```
AI-Interview/
├── README.md            # 本文件
├── docs/                # 知识点文档（markdown，按主题目录组织）
├── site-src/            # 站点样式与脚本源文件
├── build.py             # 静态站点生成器
└── .github/workflows/   # GitHub Actions 自动部署 Pages
```

## 使用方式

直接浏览文档目录即可；也欢迎通过 Issue / PR 参与补充。

## 配套网站

仓库自带一个静态站点生成器，把 `docs/` 渲染为可分类浏览、支持双链跳转、按日期追更的阅读网站：

```bash
# 本地构建并预览（生成到 site/，已被 .gitignore 忽略）
.venv/bin/python3 build.py
open site/index.html
```

推送到 `main` 分支后，GitHub Actions 会自动构建并部署到 GitHub Pages
（首次需在仓库 Settings → Pages 中将 Source 设为 **GitHub Actions**）。

## License

见仓库 LICENSE（如有）。
