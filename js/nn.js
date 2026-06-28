/* nn.js — Multi-class classification: 2 -> hidden(tanh) -> softmax MLP */
(function () {
  "use strict";
  var ML = (window.ML = window.ML || {});
  var math = ML.math;

  ML.modules = ML.modules || {};
  ML.modules.nn = function () {
    var domain = { xmin: -6, xmax: 6, ymin: -6, ymax: 6 };
    var pts = [];                 // {x,y,label}
    var H = 8, lr = 0.1, lambda = 1e-3;
    var K = 2;                    // output classes
    var seed = 12345;
    var W1, b1, W2, b2;           // params (standardized input space)
    var std = null;
    var epoch = 0, maxEpoch = 4000, lastLoss = 0;
    var dirty = true, started = false;   // started = Train has been pressed at least once

    function curK() {
      var mx = 1;
      for (var i = 0; i < pts.length; i++) if (pts[i].label > mx) mx = pts[i].label;
      return Math.max(2, mx + 1);
    }
    function zeros(n) { var a = new Array(n); for (var i = 0; i < n; i++) a[i] = 0; return a; }
    function mat(r, c, fill) { var m = []; for (var i = 0; i < r; i++) { m[i] = []; for (var j = 0; j < c; j++) m[i][j] = fill(i, j); } return m; }

    function initWeights() {
      K = curK();
      var rng = math.mulberry32(seed);
      var s1 = 1 / Math.sqrt(2), s2 = 1 / Math.sqrt(H);
      W1 = mat(2, H, function () { return math.gauss(rng) * s1; });
      b1 = zeros(H);
      W2 = mat(H, K, function () { return math.gauss(rng) * s2; });
      b2 = zeros(K);
      epoch = 0; lastLoss = 0;
    }
    function refresh() {
      std = math.standardizer(pts.map(function (p) { return [p.x, p.y]; }));
      dirty = false;
    }
    function reset() { seed = (seed * 16807 + 17) % 2147483647; initWeights(); started = false; }

    function forward(xs, ys) {
      var a1 = new Array(H), h, k;
      for (h = 0; h < H; h++) a1[h] = Math.tanh(W1[0][h] * xs + W1[1][h] * ys + b1[h]);
      var z = new Array(K), zmax = -Infinity;
      for (k = 0; k < K; k++) { var s = b2[k]; for (h = 0; h < H; h++) s += a1[h] * W2[h][k]; z[k] = s; if (s > zmax) zmax = s; }
      var p = new Array(K), sum = 0;
      for (k = 0; k < K; k++) { p[k] = Math.exp(z[k] - zmax); sum += p[k]; }
      for (k = 0; k < K; k++) p[k] /= sum;
      return { a1: a1, p: p };
    }
    function predictData(x, y) {
      if (!std) refresh();
      var f = forward((x - std.mx) / std.sx, (y - std.my) / std.sy);
      var best = 0, bv = f.p[0];
      for (var k = 1; k < K; k++) if (f.p[k] > bv) { bv = f.p[k]; best = k; }
      // confidence relative to uniform 1/K, scaled to 0..1
      var conf = (bv - 1 / K) / (1 - 1 / K);
      return { cls: best, conf: conf < 0 ? 0 : conf };
    }
    function accuracy() {
      if (pts.length === 0) return 0;
      var ok = 0;
      for (var i = 0; i < pts.length; i++) if (predictData(pts[i].x, pts[i].y).cls === pts[i].label) ok++;
      return ok / pts.length;
    }

    return {
      domain: domain,
      classCount: 4,
      presets: function () { return ML.presets.nn; },

      getXY: function () { return pts.map(function (p) { return [p.x, p.y]; }); },
      addPoint: function (x, y, label) {
        pts.push({ x: x, y: y, label: label || 0 }); dirty = true;
        if (curK() !== K) { initWeights(); started = false; }  // new class -> needs retraining
      },
      removePoint: function (i) { pts.splice(i, 1); dirty = true; },
      clear: function () { pts = []; reset(); },
      reset: reset,
      loadPreset: function (data) {
        pts = data.map(function (p) { return { x: p[0], y: p[1], label: p[2] }; });
        dirty = true; initWeights(); started = false;
      },

      setParam: function (name, v) {
        if (name === "H") { if (v !== H) { H = v; initWeights(); started = false; } }
        else if (name === "lr") lr = v;
      },

      needsTraining: function () { return pts.length >= 2; },
      onTrainStart: function () { reset(); },   // always reinit weights when Train is pressed

      step: function () {
        var n = pts.length;
        if (n < 2) return false;
        if (dirty) refresh();
        if (!W1) initWeights();
        started = true;             // training only happens on explicit Train press
        var dW1 = mat(2, H, function () { return 0; }), db1 = zeros(H);
        var dW2 = mat(H, K, function () { return 0; }), db2 = zeros(K);
        var loss = 0, i, h, k;
        for (i = 0; i < n; i++) {
          var xs = (pts[i].x - std.mx) / std.sx, ys = (pts[i].y - std.my) / std.sy;
          var f = forward(xs, ys), p = f.p, a1 = f.a1, t = pts[i].label;
          loss += -Math.log(Math.max(p[t], 1e-12));
          var dz2 = new Array(K);
          for (k = 0; k < K; k++) dz2[k] = (p[k] - (k === t ? 1 : 0)) / n;
          for (h = 0; h < H; h++) {
            var da1 = 0;
            for (k = 0; k < K; k++) { dW2[h][k] += a1[h] * dz2[k]; da1 += dz2[k] * W2[h][k]; }
            var dz1 = da1 * (1 - a1[h] * a1[h]);   // tanh'
            dW1[0][h] += xs * dz1; dW1[1][h] += ys * dz1; db1[h] += dz1;
          }
          for (k = 0; k < K; k++) db2[k] += dz2[k];
        }
        // update with L2
        for (h = 0; h < H; h++) {
          W1[0][h] -= lr * (dW1[0][h] + lambda * W1[0][h]);
          W1[1][h] -= lr * (dW1[1][h] + lambda * W1[1][h]);
          b1[h] -= lr * db1[h];
          for (k = 0; k < K; k++) W2[h][k] -= lr * (dW2[h][k] + lambda * W2[h][k]);
        }
        for (k = 0; k < K; k++) b2[k] -= lr * db2[k];
        lastLoss = loss / n; epoch++;
        if (!math.isFiniteNum(lastLoss)) { reset(); return false; }
        return epoch < maxEpoch;
      },

      render: function (plot) {
        plot.clear();
        // decision regions appear only after Train is pressed (no real-time training)
        if (started && pts.length >= 2 && W1) plot.drawRegions(predictData, 56, 0.46);
        plot.drawGrid();
        for (var i = 0; i < pts.length; i++) plot.drawPoint(pts[i].x, pts[i].y, ML.CLASS[pts[i].label]);
      },

      metrics: function () {
        var m = [];
        m.push({ label: "資料點 <span class='en'>points</span>", value: pts.length });
        m.push({ label: "類別數 <span class='en'>classes</span>", value: curK() });
        m.push({ label: "損失 <span class='en'>loss</span>", value: started ? lastLoss.toFixed(3) : "—" });
        m.push({ label: "正確率 <span class='en'>accuracy</span>", value: started ? (accuracy() * 100).toFixed(0) + "%" : "—" });
        return m;
      },

      note: function () {
        if (pts.length < 2) return "加入不同類別的點（上方切換）　<span class='en'>add points of ≥ 2 classes</span>";
        if (!started) return "按下「訓練」開始學習　<span class='en'>press Train to start</span>";
        return "";
      }
    };
  };
})();
