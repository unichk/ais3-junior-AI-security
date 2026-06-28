/* pca.js — Dimensionality reduction: 2x2 closed-form PCA + projection to PC1 */
(function () {
  "use strict";
  var ML = (window.ML = window.ML || {});

  ML.modules = ML.modules || {};
  ML.modules.pca = function () {
    var domain = { xmin: -6, xmax: 6, ymin: -6, ymax: 6 };
    var pts = [];                 // {x,y}
    var project = true;

    // returns null if degenerate, else {mean,[v1,v2],[l1,l2]}
    function compute() {
      var n = pts.length;
      if (n < 2) return null;
      var mx = 0, my = 0, i;
      for (i = 0; i < n; i++) { mx += pts[i].x; my += pts[i].y; }
      mx /= n; my /= n;
      var a = 0, d = 0, bb = 0;        // cov: [[a,bb],[bb,d]]
      for (i = 0; i < n; i++) {
        var dx = pts[i].x - mx, dy = pts[i].y - my;
        a += dx * dx; d += dy * dy; bb += dx * dy;
      }
      var den = n - 1 || 1;
      a /= den; d /= den; bb /= den;
      if (a + d < 1e-9) return null;   // all identical
      var tr = a + d;
      var rad = Math.sqrt(((a - d) / 2) * ((a - d) / 2) + bb * bb);
      var l1 = tr / 2 + rad, l2 = tr / 2 - rad;
      var v1 = eigvec(a, bb, l1);
      var v2 = [-v1[1], v1[0]];        // orthogonal
      return { mean: [mx, my], v1: v1, v2: v2, l1: l1, l2: Math.max(l2, 0) };
    }
    function eigvec(a, bb, lam) {
      var vx = bb, vy = lam - a;
      if (Math.abs(vx) < 1e-9 && Math.abs(vy) < 1e-9) { vx = 1; vy = 0; } // axis-aligned
      var nrm = Math.sqrt(vx * vx + vy * vy) || 1;
      return [vx / nrm, vy / nrm];
    }

    return {
      domain: domain,
      classCount: 1,
      lockClass: true,
      presets: function () { return ML.presets.pca; },

      getXY: function () { return pts.map(function (p) { return [p.x, p.y]; }); },
      addPoint: function (x, y) { pts.push({ x: x, y: y }); },
      removePoint: function (i) { pts.splice(i, 1); },
      clear: function () { pts = []; },
      reset: function () {},
      loadPreset: function (data) { pts = data.map(function (p) { return { x: p[0], y: p[1] }; }); },

      setParam: function (name, v) { if (name === "project") project = !!v; },
      needsTraining: function () { return false; },   // closed-form, nothing to animate
      step: function () { return false; },

      render: function (plot) {
        plot.clear(); plot.drawGrid();
        var r = compute();
        if (r) {
          var m = r.mean;
          // PC2 axis (faint) — full line through mean, direction v2; normal = v1
          plot.drawImplicitLine(r.v1[0], r.v1[1], -(r.v1[0] * m[0] + r.v1[1] * m[1]),
                                "rgba(200,186,158,0.55)", 1.5, [5, 5]);
          // projections onto PC1 (drop lines + projected points)
          if (project) {
            for (var i = 0; i < pts.length; i++) {
              var dx = pts[i].x - m[0], dy = pts[i].y - m[1];
              var t = dx * r.v1[0] + dy * r.v1[1];
              var px = m[0] + t * r.v1[0], py = m[1] + t * r.v1[1];
              plot.drawSegment([pts[i].x, pts[i].y], [px, py], "rgba(247,86,124,0.32)", 1, [3, 3]);
            }
          }
          // PC1 axis (bright) — full line through mean, direction v1; normal = (-v1y, v1x)
          plot.drawImplicitLine(-r.v1[1], r.v1[0], r.v1[1] * m[0] - r.v1[0] * m[1],
                                ML.COLORS.phos, 2.5);
        }
        // original points
        for (var j = 0; j < pts.length; j++) plot.drawPoint(pts[j].x, pts[j].y, ML.CLASS[0]);
        // projected points on top
        if (r && project) {
          for (var q = 0; q < pts.length; q++) {
            var ddx = pts[q].x - r.mean[0], ddy = pts[q].y - r.mean[1];
            var tt = ddx * r.v1[0] + ddy * r.v1[1];
            plot.drawPoint(r.mean[0] + tt * r.v1[0], r.mean[1] + tt * r.v1[1],
                           ML.CLASS[1], { r: 4, glow: 10 });
          }
        }
      },

      metrics: function () {
        var r = compute(), m = [];
        m.push({ label: "資料點 <span class='en'>points</span>", value: pts.length });
        if (r) {
          var tot = r.l1 + r.l2 || 1;
          m.push({ label: "PC1 解釋 <span class='en'>variance</span>", value: (r.l1 / tot * 100).toFixed(0) + "%" });
          m.push({ label: "PC2 解釋 <span class='en'>variance</span>", value: (r.l2 / tot * 100).toFixed(0) + "%" });
          m.push({ label: "降維後保留 <span class='en'>kept (2D→1D)</span>", value: (r.l1 / tot * 100).toFixed(1) + "%", span2: true });
        } else {
          m.push({ label: "PC1 解釋 <span class='en'>variance</span>", value: "—" });
          m.push({ label: "PC2 解釋 <span class='en'>variance</span>", value: "—" });
        }
        return m;
      },

      note: function () {
        if (pts.length < 2) return "加入至少 2 個點　<span class='en'>add ≥ 2 points</span>";
        if (!compute()) return "點不能全部重疊　<span class='en'>points must vary</span>";
        return "";
      }
    };
  };
})();
