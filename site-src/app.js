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

  /* ---------- 首页：层级索引侧边栏（分类聚焦 / 搜索 / 标签 / 滚动跟随） ---------- */
  if (page === 'home') {
    var input     = document.querySelector('.rail-search input');
    var sidebar   = document.querySelector('.side-rail');
    var toggle    = document.querySelector('.rail-toggle');
    var tree      = document.querySelector('.tree');
    var allNode   = document.querySelector('.tree-all');
    var main      = document.querySelector('.content-col');
    var emptyMsg  = document.querySelector('.filter-empty');
    var vbTitle   = document.querySelector('.vb-title');
    var vbDesc    = document.querySelector('.vb-desc');
    var vbAll     = document.querySelector('.vb-all');

    var catNodes   = Array.prototype.slice.call(document.querySelectorAll('.tree-cat'));
    var subLinks   = Array.prototype.slice.call(document.querySelectorAll('.t-sub a'));
    var tagPills   = Array.prototype.slice.call(document.querySelectorAll('.tag-pill'));
    var treeGroups = Array.prototype.slice.call(document.querySelectorAll('.tree-group'));
    var cards      = Array.prototype.slice.call(document.querySelectorAll('.category-section .article-card'));
    var sections   = Array.prototype.slice.call(document.querySelectorAll('.category-section'));
    var groups     = Array.prototype.slice.call(document.querySelectorAll('.cat-group'));

    var currentCat = '';

    function updateViewBar(shown, q) {
      if (!vbTitle || !vbDesc) return;
      var node = currentCat ? catNodes.filter(function (n) {
        return n.getAttribute('data-cat') === currentCat;
      })[0] : null;
      if (node) {
        var icon = node.querySelector('.t-icon');
        var name = node.querySelector('.t-name');
        vbTitle.textContent = (icon ? icon.textContent + ' ' : '') + (name ? name.textContent : '');
        vbDesc.textContent = 'docs/' + currentCat + '/ · 当前显示 ' + shown + ' 篇';
        if (vbAll) vbAll.hidden = false;
      } else {
        vbTitle.textContent = '🗂 全部分类';
        vbDesc.textContent = q
          ? '匹配到 ' + shown + ' 篇文章'
          : '共 ' + shown + ' 篇文章 · 点左侧分类可聚焦查看';
        if (vbAll) vbAll.hidden = true;
      }
    }

    function render() {
      var q = (input && input.value || '').trim().toLowerCase();
      var shown = 0;

      cards.forEach(function (c) {
        var sec = c.closest('.category-section');
        var catOk = !currentCat || (sec && sec.getAttribute('data-cat') === currentCat);
        var qOk = !q || (c.getAttribute('data-search') || '').toLowerCase().indexOf(q) !== -1;
        var show = catOk && qOk;
        c.style.display = show ? '' : 'none';
        if (show) shown++;
      });

      sections.forEach(function (s) {
        var vis = Array.prototype.some.call(s.querySelectorAll('.article-card'), function (c) {
          return c.style.display !== 'none';
        });
        s.style.display = vis ? '' : 'none';
      });
      groups.forEach(function (g) {
        var vis = Array.prototype.some.call(g.querySelectorAll('.category-section'), function (s) {
          return s.style.display !== 'none';
        });
        g.style.display = vis ? '' : 'none';
      });
      if (emptyMsg) emptyMsg.style.display = shown ? 'none' : '';

      /* 侧边栏的文章项随搜索同步收窄 */
      subLinks.forEach(function (a) {
        var ok = !q || (a.getAttribute('data-search') || '').toLowerCase().indexOf(q) !== -1;
        a.parentNode.style.display = ok ? '' : 'none';
      });
      catNodes.forEach(function (n) {
        var vis = Array.prototype.some.call(n.querySelectorAll('.t-sub li'), function (li) {
          return li.style.display !== 'none';
        });
        n.style.display = vis ? '' : 'none';
        n.classList.toggle('expanded', n.classList.contains('is-active') || (!!q && vis));
      });
      treeGroups.forEach(function (g) {
        var inner = g.querySelectorAll('.tree-cat');
        if (!inner.length) return;                     // 热门标签区不参与过滤
        g.style.display = Array.prototype.some.call(inner, function (n) {
          return n.style.display !== 'none';
        }) ? '' : 'none';
      });

      updateViewBar(shown, q);
    }

    function setCat(cat, scrollToMain) {
      currentCat = cat;
      catNodes.forEach(function (n) {
        var on = n.getAttribute('data-cat') === cat;
        n.classList.toggle('is-active', on);
        n.classList.remove('is-current');
        var row = n.querySelector('.t-row');
        if (row) row.setAttribute('aria-expanded', on ? 'true' : 'false');
      });
      if (allNode) allNode.classList.toggle('is-active', !cat);
      render();

      if (scrollToMain && main) {
        var top = main.getBoundingClientRect().top + window.pageYOffset - 76;
        window.scrollTo({ top: Math.max(0, top), behavior: 'smooth' });
      }
      /* 窄屏：选完分类自动收起索引，直接看内容 */
      if (sidebar && window.innerWidth <= 980) {
        sidebar.classList.remove('open');
        if (toggle) toggle.setAttribute('aria-expanded', 'false');
      }
    }

    catNodes.forEach(function (n) {
      var row = n.querySelector('.t-row');
      if (!row) return;
      row.addEventListener('click', function () {
        var cat = n.getAttribute('data-cat') || '';
        setCat(currentCat === cat ? '' : cat, true);   // 再次点击取消聚焦
      });
    });
    if (allNode) {
      allNode.addEventListener('click', function (e) { e.preventDefault(); setCat('', true); });
    }
    if (vbAll) vbAll.addEventListener('click', function () { setCat('', true); });
    if (input) input.addEventListener('input', render);

    tagPills.forEach(function (p) {
      p.addEventListener('click', function () {
        var turnOn = !p.classList.contains('is-active');
        tagPills.forEach(function (o) { o.classList.remove('is-active'); });
        if (turnOn) {
          p.classList.add('is-active');
          if (input) input.value = p.getAttribute('data-tag') || '';
        } else if (input) {
          input.value = '';
        }
        setCat('', false);
      });
    });

    if (toggle && sidebar) {
      toggle.addEventListener('click', function () {
        var open = sidebar.classList.toggle('open');
        toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
      });
    }

    /* 滚动跟随：只在「全部」视图且无搜索词时高亮当前分类 */
    function ensureVisible(el, box) {
      var top = el.offsetTop, h = el.offsetHeight;
      if (top < box.scrollTop) box.scrollTop = Math.max(0, top - 10);
      else if (top + h > box.scrollTop + box.clientHeight) {
        box.scrollTop = top + h - box.clientHeight + 10;
      }
    }
    function spy() {
      if (currentCat || (input && input.value.trim())) return;
      var best = null, bestTop = -Infinity;
      sections.forEach(function (s) {
        if (s.style.display === 'none') return;
        var top = s.getBoundingClientRect().top;
        if (top <= 150 && top > bestTop) { bestTop = top; best = s; }
      });
      var key = best ? best.getAttribute('data-cat') : '';
      catNodes.forEach(function (n) {
        var on = !!key && n.getAttribute('data-cat') === key;
        n.classList.toggle('is-current', on);
        if (on && tree) ensureVisible(n, tree);
      });
    }
    var ticking = false;
    window.addEventListener('scroll', function () {
      if (ticking) return;
      ticking = true;
      window.requestAnimationFrame(function () { spy(); ticking = false; });
    }, { passive: true });
    spy();
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
