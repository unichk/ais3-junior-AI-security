/* svm.js — Binary classification: soft-margin linear SVM (hinge subgradient) */
(function () {
  "use strict";
  var ML = (window.ML = window.ML || {});
  var math = ML.math;

  ML.modules = ML.modules || {};
  ML.modules.svm = function () {
    var domain = { xmin: -6, xmax: 6, ymin: -6, ymax: 6 };
    var pts = [];                 // {x,y,label:0|1}
    var C = 1.0;
    var lr = 0.01;
    var w = [0, 0], b = 0;        // in standardized space
    var std = null;
    var iter = 0, maxIter = 1500;
    var dirty = true;

    function refresh() {
      std = math.standardizer(pts.map(function (p) { return [p.x, p.y]; }));
      dirty = false;
    }
    function reset() { w = [0, 0]; b = 0; iter = 0; dirty = true; }
    function y1(label) { return label === 1 ? 1 : -1; }       // 0/1 -> -1/+1

    function classCounts() {
      var a = 0, c = 0;
      for (var i = 0; i < pts.length; i++) (pts[i].label === 1 ? c++ : a++);
      return [a, c];
    }
    // raw score in standardized space for a DATA point
    function score(x, y) {
      if (!std) refresh();
      var xs = (x - std.mx) / std.sx, ys = (y - std.my) / std.sy;
      return w[0] * xs + w[1] * ys + b;
    }
    function accuracy() {
      if (pts.length === 0) return 0;
      var ok = 0;
      for (var i = 0; i < pts.length; i++) {
        var s = score(pts[i].x, pts[i].y);
        if ((s >= 0 ? 1 : 0) === pts[i].label) ok++;
      }
      return ok / pts.length;
    }
    var cnt = classCounts;

    return {
      domain: domain,
      classCount: 2,
      presets: function () { return ML.presets.svm; },

      getXY: function () { return pts.map(function (p) { return [p.x, p.y]; }); },
      addPoint: function (x, y, label) { pts.push({ x: x, y: y, label: label || 0 }); dirty = true; },
      removePoint: function (i) { pts.splice(i, 1); dirty = true; },
      clear: function () { pts = []; reset(); },
      reset: reset,
      loadPreset: function (data) {
        pts = data.map(function (p) { return { x: p[0], y: p[1], label: p[2] }; });
        reset();
      },

      setParam: function (name, v) {
        if (name === "C") C = Math.pow(10, v);    // slider is log10(C)
        else if (name === "lr") lr = v;
      },

      needsTraining: function () { var c = cnt(); return c[0] > 0 && c[1] > 0; },

      step: function () {
        var c = cnt();
        if (c[0] === 0 || c[1] === 0) return false;
        if (dirty) refresh();
        var n = pts.length, gw0 = w[0], gw1 = w[1], gb = 0, i;
        // gradient of (1/2)|w|^2 + C * mean hinge
        for (i = 0; i < n; i++) {
          var xs = (pts[i].x - std.mx) / std.sx, ys = (pts[i].y - std.my) / std.sy;
          var t = y1(pts[i].label);
          var margin = t * (w[0] * xs + w[1] * ys + b);
          if (margin < 1) { gw0 -= C * t * xs / n; gw1 -= C * t * ys / n; gb -= C * t / n; }
        }
        w[0] -= lr * gw0; w[1] -= lr * gw1; b -= lr * gb;
        iter++;
        if (!math.isFiniteNum(w[0]) || !math.isFiniteNum(w[1])) { reset(); return false; }
        return iter < maxIter;
      },

      render: function (plot) {
        plot.clear();
        if (!std) refresh();
        var trained = (w[0] * w[0] + w[1] * w[1]) > 1e-9;
        // background region shading (like NN): color by predicted side, alpha by distance
        if (trained && this.needsTraining()) {
          plot.drawRegions(function (x, y) {
            var s = score(x, y);
            return { cls: s >= 0 ? 1 : 0, conf: Math.min(1, Math.abs(s)) };
          }, 56, 0.42);
        }
        plot.drawGrid();
        if (trained && this.needsTraining()) {
          // boundary & margins live in standardized space:
          //   w0*xs + w1*ys + b = k   =>  in data coords (a x + c y + d = 0)
          var a = w[0] / std.sx, cc = w[1] / std.sy;
          var base = b - w[0] * std.mx / std.sx - w[1] * std.my / std.sy;
          plot.drawImplicitLine(a, cc, base + 1, "rgba(200,186,158,0.5)", 1.5, [6, 5]);
          plot.drawImplicitLine(a, cc, base - 1, "rgba(200,186,158,0.5)", 1.5, [6, 5]);
          plot.drawImplicitLine(a, cc, base, ML.COLORS.phos, 2.5);
        }
        for (var i = 0; i < pts.length; i++) {
          var p = pts[i], opt = {};
          if (trained) {
            var t = y1(p.label), m = t * score(p.x, p.y);
            if (m <= 1 + 1e-2) opt.ring = "rgba(233,237,201,0.95)";   // support vector
          }
          plot.drawPoint(p.x, p.y, ML.CLASS[p.label], opt);
        }
      },

      metrics: function () {
        var c = cnt();
        var m = [];
        m.push({ label: "藍類 <span class='en'>class A</span>", value: c[0] });
        m.push({ label: "粉類 <span class='en'>class B</span>", value: c[1] });
        var trained = (w[0] * w[0] + w[1] * w[1]) > 1e-9 && this.needsTraining();
        m.push({ label: "正確率 <span class='en'>accuracy</span>", value: trained ? (accuracy() * 100).toFixed(0) + "%" : "—" });
        m.push({ label: "間隔 <span class='en'>margin</span>", value: trained ? (2 / Math.sqrt(w[0] * w[0] + w[1] * w[1])).toFixed(2) : "—" });
        return m;
      },

      note: function () {
        var c = cnt();
        if (c[0] === 0 || c[1] === 0) return "兩類都要有點（用上方切換類別）　<span class='en'>need both classes</span>";
        return "";
      }
    };
  };
})();
