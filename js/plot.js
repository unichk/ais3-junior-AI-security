/* plot.js — shared Canvas plotting helper (coords, axes, hit-testing, regions) */
(function () {
  "use strict";
  var ML = (window.ML = window.ML || {});

  var COLORS = {
    bg: "#15110b",
    grid: "rgba(200,186,158,0.10)",
    gridStrong: "rgba(200,186,158,0.20)",
    axis: "rgba(200,186,158,0.42)",
    tick: "rgba(196,185,163,0.55)",
    phos: "#d4a373"
  };
  // class / data colors shared with CSS (rose, sage, clay, orchid-purple)
  var CLASS = ["#f7567c", "#ccd5ae", "#d4a373", "#bf8ad6"];

  function Plot(canvas, opts) {
    opts = opts || {};
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    // fixed data domain (do not autoscale)
    this.baseDomain = opts.domain || { xmin: -6, xmax: 6, ymin: -6, ymax: 6 };
    this.domain = this.baseDomain;   // effective domain (after aspect correction)
    this.equalAspect = opts.equalAspect !== false;  // square data units by default
    this.pad = opts.pad || 34;
    this.dpr = 1;
    this.W = 10; this.H = 10;        // CSS pixel size of plot area's canvas
    this.region = null;              // offscreen ImageData-backed canvas for NN
    this.resize();
  }

  Plot.prototype.resize = function () {
    var rect = this.canvas.getBoundingClientRect();
    var cssW = Math.max(10, Math.round(rect.width));
    var cssH = Math.max(10, Math.round(rect.height));
    var dpr = window.devicePixelRatio || 1;
    this.dpr = dpr;
    this.W = cssW; this.H = cssH;
    this.canvas.width = Math.round(cssW * dpr);
    this.canvas.height = Math.round(cssH * dpr);
    var ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // draw in CSS pixels
    this._applyAspect();
  };

  // expand the shorter axis so 1 data unit == same #px on x and y (geometry correct)
  Plot.prototype._applyAspect = function () {
    var bd = this.baseDomain;
    if (!this.equalAspect) { this.domain = bd; return; }
    var pw = this.plotW(), ph = this.plotH();
    var spanX = bd.xmax - bd.xmin, spanY = bd.ymax - bd.ymin;
    var cx = (bd.xmin + bd.xmax) / 2, cy = (bd.ymin + bd.ymax) / 2;
    var scale = Math.min(pw / spanX, ph / spanY);   // px per data unit
    var ex = pw / scale, ey = ph / scale;
    this.domain = { xmin: cx - ex / 2, xmax: cx + ex / 2, ymin: cy - ey / 2, ymax: cy + ey / 2 };
  };

  // ---- coordinate transforms (data <-> CSS pixels) ----
  Plot.prototype.plotW = function () { return this.W - 2 * this.pad; };
  Plot.prototype.plotH = function () { return this.H - 2 * this.pad; };
  Plot.prototype.dataToPx = function (x, y) {
    var d = this.domain;
    var px = this.pad + ((x - d.xmin) / (d.xmax - d.xmin)) * this.plotW();
    var py = this.pad + (1 - (y - d.ymin) / (d.ymax - d.ymin)) * this.plotH();
    return [px, py];
  };
  Plot.prototype.pxToData = function (px, py) {
    var d = this.domain;
    var x = d.xmin + ((px - this.pad) / this.plotW()) * (d.xmax - d.xmin);
    var y = d.ymin + (1 - (py - this.pad) / this.plotH()) * (d.ymax - d.ymin);
    return [x, y];
  };
  // mouse event -> data coords
  Plot.prototype.eventToData = function (ev) {
    var rect = this.canvas.getBoundingClientRect();
    return this.pxToData(ev.clientX - rect.left, ev.clientY - rect.top);
  };

  // hit-test points (array of [x,y]); returns index within px radius, else -1
  Plot.prototype.hitTest = function (px, py, points, radiusPx) {
    radiusPx = radiusPx || 9;
    var best = -1, bestD = radiusPx * radiusPx;
    for (var i = 0; i < points.length; i++) {
      var p = this.dataToPx(points[i][0], points[i][1]);
      var dx = p[0] - px, dy = p[1] - py, d = dx * dx + dy * dy;
      if (d <= bestD) { bestD = d; best = i; }
    }
    return best;
  };

  // ---- drawing primitives ----
  Plot.prototype.clear = function () {
    var ctx = this.ctx;
    ctx.clearRect(0, 0, this.W, this.H);
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, this.W, this.H);
  };

  Plot.prototype.drawGrid = function () {
    var ctx = this.ctx, d = this.domain;
    var step = niceStep((d.xmax - d.xmin) / 8);
    ctx.lineWidth = 1;
    ctx.font = "10px 'Space Mono', monospace";
    ctx.fillStyle = COLORS.tick;
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    var x;
    for (x = Math.ceil(d.xmin / step) * step; x <= d.xmax + 1e-9; x += step) {
      var pa = this.dataToPx(x, d.ymin), pb = this.dataToPx(x, d.ymax);
      ctx.strokeStyle = Math.abs(x) < 1e-9 ? COLORS.axis : COLORS.grid;
      ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
      if (Math.abs(x) > 1e-9) ctx.fillText(fmt(x), pa[0], this.H - this.pad + 5);
    }
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    var y;
    for (y = Math.ceil(d.ymin / step) * step; y <= d.ymax + 1e-9; y += step) {
      var qa = this.dataToPx(d.xmin, y), qb = this.dataToPx(d.xmax, y);
      ctx.strokeStyle = Math.abs(y) < 1e-9 ? COLORS.axis : COLORS.grid;
      ctx.beginPath(); ctx.moveTo(qa[0], qa[1]); ctx.lineTo(qb[0], qb[1]); ctx.stroke();
      if (Math.abs(y) > 1e-9) ctx.fillText(fmt(y), this.pad - 6, qa[1]);
    }
  };

  // point with glow; color index or css color
  Plot.prototype.drawPoint = function (x, y, color, opts) {
    opts = opts || {};
    var ctx = this.ctx, p = this.dataToPx(x, y);
    var r = opts.r || 5;
    ctx.save();
    if (opts.ring) {
      ctx.strokeStyle = opts.ring; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(p[0], p[1], r + 4, 0, 7); ctx.stroke();
    }
    ctx.shadowColor = color; ctx.shadowBlur = opts.glow == null ? 8 : opts.glow;
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 7); ctx.fill();
    ctx.shadowBlur = 0;
    ctx.lineWidth = 1.2; ctx.strokeStyle = "rgba(0,0,0,0.55)";
    ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 7); ctx.stroke();
    ctx.restore();
  };

  // line through two data points
  Plot.prototype.drawSegment = function (a, b, color, width, dash) {
    var ctx = this.ctx, pa = this.dataToPx(a[0], a[1]), pb = this.dataToPx(b[0], b[1]);
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = width || 2;
    if (dash) ctx.setLineDash(dash);
    ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
    ctx.restore();
  };

  // draw an infinite-ish line a*x + b*y + c = 0 clipped to domain
  Plot.prototype.drawImplicitLine = function (a, b, c, color, width, dash) {
    var d = this.domain, pts = [];
    // intersect with the 4 borders
    if (Math.abs(b) > 1e-12) {
      pts.push([d.xmin, -(a * d.xmin + c) / b]);
      pts.push([d.xmax, -(a * d.xmax + c) / b]);
    }
    if (Math.abs(a) > 1e-12) {
      pts.push([-(b * d.ymin + c) / a, d.ymin]);
      pts.push([-(b * d.ymax + c) / a, d.ymax]);
    }
    var inside = pts.filter(function (p) {
      return p[0] >= d.xmin - 1e-6 && p[0] <= d.xmax + 1e-6 &&
             p[1] >= d.ymin - 1e-6 && p[1] <= d.ymax + 1e-6;
    });
    if (inside.length >= 2) this.drawSegment(inside[0], inside[1], color, width, dash);
  };

  // ---- decision-region rendering via small offscreen ImageData ----
  // predictFn(x,y) -> { cls: int, conf: 0..1 }
  Plot.prototype.drawRegions = function (predictFn, gridN, alphaMax) {
    gridN = gridN || 56;
    alphaMax = alphaMax == null ? 0.5 : alphaMax;
    if (!this.region || this.region.n !== gridN) {
      var oc = document.createElement("canvas");
      oc.width = gridN; oc.height = gridN;
      this.region = { canvas: oc, ctx: oc.getContext("2d"), n: gridN,
                      img: oc.getContext("2d").createImageData(gridN, gridN) };
    }
    var R = this.region, img = R.img, data = img.data, d = this.domain;
    var k = 0;
    for (var gy = 0; gy < gridN; gy++) {
      var fy = 1 - (gy + 0.5) / gridN;          // top row = ymax
      var yy = d.ymin + fy * (d.ymax - d.ymin);
      for (var gx = 0; gx < gridN; gx++) {
        var fx = (gx + 0.5) / gridN;
        var xx = d.xmin + fx * (d.xmax - d.xmin);
        var pr = predictFn(xx, yy);
        var col = hexToRgb(CLASS[pr.cls % CLASS.length]);
        data[k]   = col[0];
        data[k+1] = col[1];
        data[k+2] = col[2];
        data[k+3] = Math.round(alphaMax * 255 * (0.35 + 0.65 * pr.conf));
        k += 4;
      }
    }
    R.ctx.putImageData(img, 0, 0);
    var ctx = this.ctx;
    ctx.save();
    ctx.imageSmoothingEnabled = true;
    // map the data domain rectangle onto pixel rectangle
    var tl = this.dataToPx(d.xmin, d.ymax);
    var br = this.dataToPx(d.xmax, d.ymin);
    ctx.drawImage(R.canvas, tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]);
    ctx.restore();
  };

  // ---- helpers ----
  function niceStep(raw) {
    var p = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
    var f = raw / p;
    var nf = f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10;
    return nf * p;
  }
  function fmt(v) {
    if (Math.abs(v) >= 1000) return (v / 1000) + "k";
    return Math.round(v * 100) / 100 + "";
  }
  function hexToRgb(h) {
    var n = parseInt(h.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  ML.Plot = Plot;
  ML.COLORS = COLORS;
  ML.CLASS = CLASS;
})();
