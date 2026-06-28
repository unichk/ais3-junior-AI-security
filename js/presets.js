/* presets.js — preset datasets (generated deterministically; file:// safe, no fetch) */
(function () {
  "use strict";
  var ML = (window.ML = window.ML || {});
  var rng = ML.math.mulberry32(987654321);
  function g() { return ML.math.gauss(rng); }

  function blob(cx, cy, sd, n, label) {
    var a = [];
    for (var i = 0; i < n; i++) a.push([cx + g() * sd, cy + g() * sd, label]);
    return a;
  }
  // ring of points at radius r (with noise)
  function ring(r, sd, n, label) {
    var a = [];
    for (var i = 0; i < n; i++) {
      var th = (i / n) * Math.PI * 2;
      a.push([r * Math.cos(th) + g() * sd, r * Math.sin(th) + g() * sd, label]);
    }
    return a;
  }
  // two interleaving half-moons
  function moons(n, R, sd) {
    var a = [];
    for (var i = 0; i < n; i++) {
      var t = Math.PI * (i / (n - 1));
      a.push([R * Math.cos(t) - 1.5 + g() * sd, R * Math.sin(t) - 1.0 + g() * sd, 0]);
      a.push([R * Math.cos(t) + 1.5 + g() * sd, -R * Math.sin(t) + 1.0 + g() * sd, 1]);
    }
    return a;
  }

  // ---- Linear Regression: [x, y], domain ~ x[0,20] y[2,14] ----
  function line(slope, intercept, sd, n) {
    var a = [];
    for (var i = 0; i < n; i++) {
      var x = 3 + (17 - 3) * (i / (n - 1));
      a.push([x, slope * x + intercept + g() * sd]);
    }
    return a;
  }
  // Anscombe's quartet — four sets with (almost) identical best-fit lines
  function zip(xs, ys) { return xs.map(function (x, i) { return [x, ys[i]]; }); }
  var ANS_X = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5];

  // ---- spiral for NN ----
  function spiral(turns, n, label, phase) {
    var a = [];
    for (var i = 0; i < n; i++) {
      var r = 0.4 + (i / n) * 5;
      var th = (i / n) * turns * Math.PI * 2 + phase;
      a.push([r * Math.cos(th) + g() * 0.18, r * Math.sin(th) + g() * 0.18, label]);
    }
    return a;
  }

  // tilted gaussian cloud for PCA: major sd, minor sd, rotation
  function tilt(major, minor, rot, n) {
    var c = Math.cos(rot), s = Math.sin(rot), a = [];
    for (var i = 0; i < n; i++) {
      var u = g() * major, v = g() * minor;
      a.push([u * c - v * s, u * s + v * c]);
    }
    return a;
  }

  ML.presets = {
    linreg: [
      { name: "安士庫姆 I · 線性 (Anscombe I)",
        data: zip(ANS_X, [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]) },
      { name: "安士庫姆 II · 曲線 (Anscombe II)",
        data: zip(ANS_X, [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]) },
      { name: "安士庫姆 III · 離群值 (Anscombe III)",
        data: zip(ANS_X, [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]) },
      { name: "安士庫姆 IV · 垂直 (Anscombe IV)",
        data: zip([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
                  [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]) },
      { name: "緊密擬合 (tight)", data: line(0.5, 3, 0.3, 16) },
      { name: "平緩斜率 (gentle)", data: line(0.22, 5, 0.5, 14) }
    ],
    svm: [
      { name: "可分開 (separable)", data: blob(-3, -2.2, 0.8, 14, 0).concat(blob(2.8, 2.3, 0.8, 14, 1)) },
      { name: "略重疊 (overlapping)", data: blob(-1.6, -1, 1.3, 18, 0).concat(blob(1.6, 1, 1.3, 18, 1)) },
      { name: "對角 (diagonal)", data: blob(-3, 2.5, 1.0, 14, 0).concat(blob(2.5, -2.8, 1.0, 14, 1)) },
      { name: "垂直分離 (vertical)", data: blob(-2.6, 0, 0.9, 16, 0).concat(blob(2.6, 0, 0.9, 16, 1)) },
      { name: "水平分離 (horizontal)", data: blob(0, 2.6, 0.9, 16, 0).concat(blob(0, -2.6, 0.9, 16, 1)) },
      { name: "高度重疊 (hard)", data: blob(-0.9, 0, 1.5, 20, 0).concat(blob(0.9, 0, 1.5, 20, 1)) }
    ],
    nn: [
      { name: "三群 (3 clusters)",
        data: blob(0, 3.2, 0.7, 12, 0).concat(blob(-3, -2.2, 0.7, 12, 1)).concat(blob(3, -2.2, 0.7, 12, 2)) },
      { name: "四象限 (4 quadrants)",
        data: blob(2.6, 2.6, 0.7, 9, 0).concat(blob(-2.6, 2.6, 0.7, 9, 1))
              .concat(blob(-2.6, -2.6, 0.7, 9, 2)).concat(blob(2.6, -2.6, 0.7, 9, 3)) },
      { name: "雙螺旋 (two spirals)",
        data: spiral(1.0, 40, 0, 0).concat(spiral(1.0, 40, 1, Math.PI)) },
      { name: "同心圓 (circles)",
        data: blob(0, 0, 0.6, 16, 0).concat(ring(4, 0.35, 28, 1)) },
      { name: "兩月牙 (two moons)", data: moons(24, 3, 0.28) },
      { name: "四臂螺旋 (4-arm spiral)",
        data: spiral(0.7, 22, 0, 0).concat(spiral(0.7, 22, 1, Math.PI / 2))
              .concat(spiral(0.7, 22, 2, Math.PI)).concat(spiral(0.7, 22, 3, 3 * Math.PI / 2)) }
    ],
    pca: [
      { name: "斜向雲 (diagonal)", data: tilt(2.6, 0.6, 0.7, 40) },
      { name: "近似一維 (nearly 1D)", data: tilt(3.0, 0.18, -0.5, 40) },
      { name: "接近圓形 (isotropic)", data: tilt(1.8, 1.7, 0.3, 40) },
      { name: "水平延展 (horizontal)", data: tilt(3.2, 0.7, 0, 40) },
      { name: "垂直延展 (vertical)", data: tilt(3.2, 0.7, Math.PI / 2, 40) },
      { name: "反斜向 (anti-diagonal)", data: tilt(2.6, 0.6, -0.7, 40) }
    ]
  };
})();
