/* AI Interview 知识库 · 交互脚本（纯增强，无 JS 时页面完整可读） */
(function () {
  'use strict';

  var page = document.body.getAttribute('data-page') || '';

  /* ---------- 文章页：阅读进度条 / 目录高亮 / 返回顶部 ---------- */
  if (page === 'article') {
    var bar = document.querySelector('.progress-bar');
    var backTop = document.querySelector('.back-top');

    function onScroll() {
      var doc = document.documentElement;
      var total = doc.scrollHeight - window.innerHeight;
      var pct = total > 0 ? (window.scrollY / total) * 100 : 0;
      if (bar) bar.style.width = Math.min(100, pct) + '%';
      if (backTop) backTop.classList.toggle('show', window.scrollY > 480);
    }
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();

    if (backTop) {
      backTop.addEventListener('click', function () {
        window.scrollTo({ top: 0, behavior: 'smooth' });
      });
    }

    /* 目录 scroll-spy */
    var tocLinks = Array.prototype.slice.call(document.querySelectorAll('.toc-sidebar a[href^="#"]'));
    if (tocLinks.length && 'IntersectionObserver' in window) {
      var map = {};
      tocLinks.forEach(function (a) {
        var id = decodeURIComponent(a.getAttribute('href').slice(1));
        var el = document.getElementById(id);
        if (el) map[id] = a;
      });
      var visible = {};
      var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) { visible[e.target.id] = e.isIntersecting; });
        /* 取当前可见的最靠上的标题；都不可见时取最近经过的 */
        var best = null;
        Object.keys(map).forEach(function (id) {
          var top = document.getElementById(id).getBoundingClientRect().top;
          if (top <= 120) best = best && document.getElementById(best).getBoundingClientRect().top > top ? best : id;
        });
        tocLinks.forEach(function (a) { a.classList.remove('active'); });
        if (best && map[best]) map[best].classList.add('active');
      }, { rootMargin: '-70px 0px -55% 0px', threshold: 0 });
      Object.keys(map).forEach(function (id) { observer.observe(document.getElementById(id)); });
    }

    /* 窄屏目录折叠 */
    var tocBox = document.querySelector('.toc-sidebar');
    var tocTitle = document.querySelector('.toc-sidebar .toc-title');
    if (tocBox && tocTitle && window.innerWidth <= 980) {
      tocTitle.addEventListener('click', function () { tocBox.classList.toggle('expanded'); });
    } else if (tocBox) {
      tocBox.classList.add('expanded');
    }
  }

  /* ---------- 首页：按标题 / 标签 / 分类过滤 ---------- */
  if (page === 'home') {
    var input = document.querySelector('.filter-bar input');
    var cards = Array.prototype.slice.call(document.querySelectorAll('.article-card'));
    var sections = Array.prototype.slice.call(document.querySelectorAll('.category-section'));
    var noResult = document.querySelector('.filter-empty');

    function applyFilter() {
      var q = (input.value || '').trim().toLowerCase();
      var anyVisible = false;
      cards.forEach(function (c) {
        var hay = (c.getAttribute('data-search') || '').toLowerCase();
        var show = !q || hay.indexOf(q) !== -1;
        c.style.display = show ? '' : 'none';
        if (show) anyVisible = true;
      });
      sections.forEach(function (s) {
        var visible = Array.prototype.some.call(
          s.querySelectorAll('.article-card'),
          function (c) { return c.style.display !== 'none'; }
        );
        s.style.display = visible ? '' : 'none';
      });
      if (noResult) noResult.style.display = anyVisible ? 'none' : '';
    }
    if (input) input.addEventListener('input', applyFilter);
  }

  /* ---------- 最近更新页：按日期筛选 ---------- */
  if (page === 'updates') {
    var picker = document.getElementById('date-picker');
    var groups = Array.prototype.slice.call(document.querySelectorAll('.day-group'));
    var clearBtn = document.querySelector('.btn-clear');
    var emptyMsg = document.querySelector('.filter-empty');
    var chips = Array.prototype.slice.call(document.querySelectorAll('.date-chip'));

    function applyDate(dateStr) {
      var anyVisible = false;
      groups.forEach(function (g) {
        var show = !dateStr || g.getAttribute('data-date') === dateStr;
        g.style.display = show ? '' : 'none';
        if (show) anyVisible = true;
      });
      if (emptyMsg) emptyMsg.style.display = anyVisible ? 'none' : '';
      chips.forEach(function (ch) {
        ch.classList.toggle('active', ch.getAttribute('data-date') === dateStr);
      });
    }

    if (picker) {
      picker.addEventListener('change', function () { applyDate(picker.value); });
    }
    chips.forEach(function (ch) {
      ch.addEventListener('click', function () {
        var d = ch.getAttribute('data-date');
        if (picker) picker.value = d;
        applyDate(d);
      });
    });
    if (clearBtn) {
      clearBtn.addEventListener('click', function () {
        if (picker) picker.value = '';
        applyDate('');
      });
    }
  }
})();
