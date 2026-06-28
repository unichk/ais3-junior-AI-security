/* linreg.js — Linear Regression: closed-form least squares (instant, no lr) */
(function () {
  "use strict";
  var ML = (window.ML = window.ML || {});

  ML.modules = ML.modules || {};
  ML.modules.linreg = function () {
    var domain = { xmin: 0, xmax: 20, ymin: 2, ymax: 14 };
    var pts = [];                 // {x,y}

    // closed-form least squares in data space; null if degenerate
    function fit() {
      var n = pts.length; if (n < 2) return null;
      var mx = 0, my = 0, i;
      for (i = 0; i < n; i++) { mx += pts[i].x; my += pts[i].y; }
      mx /= n; my /= n;
      var sxy = 0, sxx = 0;
      for (i = 0; i < n; i++) { sxy += (pts[i].x - mx) * (pts[i].y - my); sxx += (pts[i].x - mx) * (pts[i].x - mx); }
      if (sxx < 1e-9) return null;             // vertical / identical x
      var w = sxy / sxx;
      return { w: w, b: my - w * mx };
    }
    function mse(f) {
      if (!f || pts.length === 0) return 0;
      var s = 0;
      for (var i = 0; i < pts.length; i++) { var e = (f.w * pts[i].x + f.b) - pts[i].y; s += e * e; }
      return s / pts.length;
    }

    return {
      domain: domain,
      equalAspect: false,        // x and y are different quantities; fill the plot
      classCount: 1,
      lockClass: true,
      presets: function () { return ML.presets.linreg; },

      getXY: function () { return pts.map(function (p) { return [p.x, p.y]; }); },
      addPoint: function (x, y) { pts.push({ x: x, y: y }); },
      removePoint: function (i) { pts.splice(i, 1); },
      clear: function () { pts = []; },
      reset: function () {},
      loadPreset: function (data) { pts = data.map(function (p) { return { x: p[0], y: p[1] }; }); },

      setParam: function () {},
      needsTraining: function () { return false; },  // closed-form is instant
      step: function () { return false; },

      render: function (plot) {
        plot.clear(); plot.drawGrid();
        var f = fit();
        if (f) {
          // residuals (point -> line)
          for (var i = 0; i < pts.length; i++) {
            plot.drawSegment([pts[i].x, pts[i].y], [pts[i].x, f.w * pts[i].x + f.b],
                             "rgba(247,86,124,0.42)", 1);
          }
          // best-fit line (closed-form solution)
          plot.drawSegment([domain.xmin, f.w * domain.xmin + f.b],
                           [domain.xmax, f.w * domain.xmax + f.b],
                           ML.COLORS.phos, 2.6);
        }
        for (var j = 0; j < pts.length; j++) plot.drawPoint(pts[j].x, pts[j].y, ML.CLASS[0]);
      },

      metrics: function () {
        var f = fit();
        return [
          { label: "資料點 <span class='en'>points</span>", value: pts.length },
          { label: "誤差 <span class='en'>MSE</span>", value: f ? mse(f).toFixed(3) : "—" },
          { label: "斜率 <span class='en'>w</span>", value: f ? f.w.toFixed(3) : "—" },
          { label: "截距 <span class='en'>b</span>", value: f ? f.b.toFixed(3) : "—" }
        ];
      },

      note: function () {
        if (pts.length < 2) return "點按圖面加入至少 2 個點　<span class='en'>add ≥ 2 points</span>";
        if (!fit()) return "需要 x 不同的點　<span class='en'>need varied x values</span>";
        return "";
      }
    };
  };
})();
