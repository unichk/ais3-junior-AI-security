/* mathutils.js — shared numeric helpers (file:// safe, global namespace) */
(function () {
  "use strict";
  var ML = (window.ML = window.ML || {});

  // Seeded PRNG so "reset" is reproducible. Returns fn -> [0,1).
  function mulberry32(seed) {
    var a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Standard normal via Box–Muller, driven by a [0,1) rng.
  function gauss(rng) {
    var u = 1 - rng(), v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // Per-feature mean/std over array of [x,y]. std guarded to >= 1e-8.
  function standardizer(points) {
    var n = points.length || 1;
    var mx = 0, my = 0;
    for (var i = 0; i < points.length; i++) { mx += points[i][0]; my += points[i][1]; }
    mx /= n; my /= n;
    var vx = 0, vy = 0;
    for (var j = 0; j < points.length; j++) {
      vx += (points[j][0] - mx) * (points[j][0] - mx);
      vy += (points[j][1] - my) * (points[j][1] - my);
    }
    var sx = Math.sqrt(vx / n) || 0, sy = Math.sqrt(vy / n) || 0;
    if (sx < 1e-8) sx = 1;
    if (sy < 1e-8) sy = 1;
    return {
      mx: mx, my: my, sx: sx, sy: sy,
      fwd: function (p) { return [(p[0] - mx) / sx, (p[1] - my) / sy]; }
    };
  }

  function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

  function isFiniteNum(v) { return typeof v === "number" && isFinite(v); }

  ML.math = {
    mulberry32: mulberry32,
    gauss: gauss,
    standardizer: standardizer,
    clamp: clamp,
    isFiniteNum: isFiniteNum
  };
})();
