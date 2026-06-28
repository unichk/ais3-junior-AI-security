/* app.js — tab lifecycle, control wiring, render loop (loaded last) */
(function () {
  "use strict";
  var ML = window.ML;
  var KEYS = ["linreg", "svm", "nn", "pca"];
  var CLASS_ZH = ["藍 A", "粉 B", "金 C", "紫 D"];

  var controllers = {};
  var activeKey = null;

  // mini preview of a preset dataset, framed by the module's full canvas domain
  function drawThumb(canvas, data, domain) {
    var ctx = canvas.getContext("2d"), W = canvas.width, H = canvas.height, pad = 6;
    ctx.fillStyle = "#15110b"; ctx.fillRect(0, 0, W, H);
    if (!data || !data.length) return;
    var d = domain || { xmin: -6, xmax: 6, ymin: -6, ymax: 6 };
    function px(x) { return pad + (x - d.xmin) / (d.xmax - d.xmin) * (W - 2 * pad); }
    function py(y) { return pad + (1 - (y - d.ymin) / (d.ymax - d.ymin)) * (H - 2 * pad); }
    data.forEach(function (p) {
      var col = ML.CLASS[p.length > 2 ? p[2] : 0];
      ctx.fillStyle = col; ctx.shadowColor = col; ctx.shadowBlur = 4;
      ctx.beginPath(); ctx.arc(px(p[0]), py(p[1]), 2.6, 0, 7); ctx.fill();
    });
    ctx.shadowBlur = 0;
  }

  function fmtParam(name, v) {
    if (name === "C") return Math.pow(10, v).toFixed(2);
    if (name === "H") return String(v | 0);
    return Number(v).toFixed(2);
  }

  function makeController(key) {
    var module = ML.modules[key]();
    var canvas = document.getElementById("canvas-" + key);
    var panel = document.getElementById("panel-" + key);
    var controlsEl = document.getElementById("controls-" + key);
    var metricsEl = document.getElementById("metrics-" + key);
    var noteEl = document.getElementById("note-" + key);
    var playBtn = controlsEl.querySelector('[data-act="play"]');
    var plot = null, raf = null, playing = false, needsRender = true, currentClass = 0;

    // --- class picker ---
    var picker = controlsEl.querySelector("[data-classpicker]");
    if (picker && module.classCount > 1) {
      for (var ci = 0; ci < module.classCount; ci++) {
        (function (idx) {
          var chip = document.createElement("button");
          chip.className = "class-chip" + (idx === 0 ? " is-active" : "");
          chip.style.color = ML.CLASS[idx];
          chip.innerHTML = '<span class="dot" style="background:' + ML.CLASS[idx] + '"></span>' + CLASS_ZH[idx];
          chip.addEventListener("click", function () {
            currentClass = idx;
            picker.querySelectorAll(".class-chip").forEach(function (c) { c.classList.remove("is-active"); });
            chip.classList.add("is-active");
          });
          picker.appendChild(chip);
        })(ci);
      }
    }

    // --- preset gallery (visual thumbnails) ---
    var gallery = controlsEl.querySelector("[data-presets]");
    if (gallery) {
      module.presets().forEach(function (p) {
        var item = document.createElement("button");
        item.className = "preset-item";
        var cv = document.createElement("canvas");
        cv.width = 132; cv.height = 78;
        item.appendChild(cv);
        var cap = document.createElement("span");
        cap.className = "preset-cap"; cap.textContent = p.name;
        item.appendChild(cap);
        drawThumb(cv, p.data, module.domain);
        item.addEventListener("click", function () {
          module.loadPreset(p.data);
          gallery.querySelectorAll(".preset-item").forEach(function (x) { x.classList.remove("is-active"); });
          item.classList.add("is-active");
          needsRender = true;
        });
        gallery.appendChild(item);
      });
    }

    // --- sliders ---
    controlsEl.querySelectorAll('input[type="range"]').forEach(function (inp) {
      var name = inp.getAttribute("data-param");
      var label = controlsEl.querySelector('[data-val="' + name + '"]');
      function apply() {
        var v = parseFloat(inp.value);
        if (label) label.textContent = fmtParam(name, v);
        module.setParam(name, name === "H" ? (v | 0) : v);
        needsRender = true;
      }
      inp.addEventListener("input", apply);
      apply(); // initialize module param from default
    });

    // --- checkbox params ---
    controlsEl.querySelectorAll('input[type="checkbox"][data-param]').forEach(function (inp) {
      var name = inp.getAttribute("data-param");
      function apply() { module.setParam(name, inp.checked); needsRender = true; }
      inp.addEventListener("change", apply);
      apply();
    });

    // --- buttons ---
    function setPlay(on) {
      playing = on;
      if (!playBtn) return;
      playBtn.classList.toggle("is-running", on);
      playBtn.innerHTML = on ? '⏸ 暫停 <span class="en">Pause</span>' : '▶ 訓練 <span class="en">Train</span>';
    }
    controlsEl.querySelectorAll("[data-act]").forEach(function (btn) {
      var act = btn.getAttribute("data-act");
      btn.addEventListener("click", function () {
        if (act === "play") {
          if (!module.needsTraining()) { needsRender = true; return; }
          if (!playing && module.onTrainStart) module.onTrainStart();  // fresh start on each Train
          setPlay(!playing);
        } else if (act === "reset") {
          module.reset(); needsRender = true;
        } else if (act === "clear") {
          setPlay(false); module.clear(); needsRender = true;
        }
      });
    });

    // --- canvas interaction ---
    canvas.addEventListener("click", function (ev) {
      ensurePlot();
      var d = plot.eventToData(ev);
      module.addPoint(d[0], d[1], currentClass);
      needsRender = true;
    });
    canvas.addEventListener("contextmenu", function (ev) {
      ev.preventDefault();
      ensurePlot();
      var rect = canvas.getBoundingClientRect();
      var hit = plot.hitTest(ev.clientX - rect.left, ev.clientY - rect.top, module.getXY());
      if (hit >= 0) { module.removePoint(hit); needsRender = true; }
    });

    function ensurePlot() {
      if (!plot) plot = new ML.Plot(canvas, { domain: module.domain, equalAspect: module.equalAspect !== false });
    }
    function resize() { if (plot) { plot.resize(); needsRender = true; } }

    function renderMetrics() {
      var arr = module.metrics(), html = "";
      for (var i = 0; i < arr.length; i++) {
        html += '<div class="metric' + (arr[i].span2 ? " span2" : "") + '">' +
                '<span class="m-label">' + arr[i].label + '</span>' +
                '<span class="m-value">' + arr[i].value + "</span></div>";
      }
      metricsEl.innerHTML = html;
    }
    function renderNote() {
      var t = module.note();
      if (t) { noteEl.innerHTML = t; noteEl.classList.add("show"); }
      else noteEl.classList.remove("show");
    }

    function loop() {
      if (activeKey !== key) { raf = null; return; }
      if (playing) {
        var t0 = performance.now(), more = true;
        while (more && performance.now() - t0 < 8) more = module.step();
        needsRender = true;
        if (!more) setPlay(false);
      }
      if (needsRender) {
        ensurePlot();
        module.render(plot);
        renderMetrics();
        renderNote();
        needsRender = false;
      }
      raf = requestAnimationFrame(loop);
    }

    return {
      activate: function () {
        panel.hidden = false; panel.classList.add("is-active");
        ensurePlot(); plot.resize();
        needsRender = true;
        if (!raf) raf = requestAnimationFrame(loop);
      },
      deactivate: function () {
        panel.hidden = true; panel.classList.remove("is-active");
        if (raf) { cancelAnimationFrame(raf); raf = null; }
      },
      resize: resize
    };
  }

  function switchTab(key) {
    if (key === activeKey) return;
    document.querySelectorAll(".tab").forEach(function (t) {
      var on = t.getAttribute("data-tab") === key;
      t.classList.toggle("is-active", on);
      t.setAttribute("aria-selected", on ? "true" : "false");
    });
    if (activeKey && controllers[activeKey]) controllers[activeKey].deactivate();
    activeKey = key;
    controllers[key].activate();
  }

  document.addEventListener("DOMContentLoaded", function () {
    KEYS.forEach(function (k) { controllers[k] = makeController(k); });
    document.querySelectorAll(".tab").forEach(function (t) {
      t.addEventListener("click", function () { switchTab(t.getAttribute("data-tab")); });
    });
    var resizeT = null;
    window.addEventListener("resize", function () {
      clearTimeout(resizeT);
      resizeT = setTimeout(function () { if (activeKey) controllers[activeKey].resize(); }, 120);
    });
    // activate first tab
    activeKey = null;
    switchTab("linreg");
  });
})();
