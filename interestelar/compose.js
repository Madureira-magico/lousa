// Monta a imagem parada do relógio (com ou sem ponteiros) com a mesma matemática do Python.
'use strict';
const Compose = (() => {
  let M, W, H, X0, Y0, base, gmul, gadd, spr = {}, ready = false;

  function imgData(img) {
    const c = new OffscreenCanvas(img.width, img.height);
    const x = c.getContext('2d', { willReadFrequently: true }); x.drawImage(img, 0, 0);
    return x.getImageData(0, 0, img.width, img.height);
  }
  async function loadImg(src) {
    const r = await fetch(src); if (!r.ok) throw new Error(src);
    return createImageBitmap(await r.blob(), { premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
  }
  async function init(meta, dir) {
    M = meta; [X0, Y0] = meta.bbox; W = meta.bbox[2] - X0; H = meta.bbox[3] - Y0;
    const [b, gm, ga, h1, h2, h3] = await Promise.all(['base', 'gmul', 'gadd', 'hand_hour', 'hand_min', 'hand_sec'].map(n => loadImg(dir + n + '.png')));
    base = imgData(b).data; gmul = imgData(gm).data; gadd = imgData(ga).data;
    for (const [k, im] of [['hour', h1], ['min', h2], ['sec', h3]]) { const d = imgData(im); spr[k] = { w: d.width, h: d.height, d: d.data }; }
    ready = true;
  }

  // ---------- gaussian blur (separable, BORDER_REFLECT), single channel Float32 ----------
  function gaussKernel(s) {
    const r = Math.max(1, Math.round(4 * s)), k = new Float32Array(2 * r + 1); let sum = 0;
    for (let i = -r; i <= r; i++) { const v = Math.exp(-(i * i) / (2 * s * s)); k[i + r] = v; sum += v; }
    for (let i = 0; i < k.length; i++) k[i] /= sum; return { r, k };
  }
  const refl = (i, n) => { while (i < 0 || i >= n) { if (i < 0) i = -i - 1; if (i >= n) i = 2 * n - i - 1; } return i; };
  function blur1(src, w, h, s) {
    if (s <= 0.05) return src;
    const { r, k } = gaussKernel(s), tmp = new Float32Array(w * h), out = new Float32Array(w * h);
    for (let y = 0; y < h; y++) { const row = y * w;
      for (let x = 0; x < w; x++) { let a = 0;
        for (let j = -r; j <= r; j++) a += src[row + refl(x + j, w)] * k[j + r]; tmp[row + x] = a; } }
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { let a = 0;
      for (let j = -r; j <= r; j++) a += tmp[refl(y + j, h) * w + x] * k[j + r]; out[y * w + x] = a; }
    return out;
  }
  // fast big blur (3 box passes ≈ gaussian), used for the bloom (same for every state, so it cancels)
  function boxBlur(src, w, h, s) {
    const n = 3, wIdeal = Math.sqrt((12 * s * s / n) + 1); let wl = Math.floor(wIdeal); if (wl % 2 === 0) wl--;
    const m = Math.round((12 * s * s - n * wl * wl - 4 * n * wl - 3 * n) / (-4 * wl - 4));
    let a = Float32Array.from(src), b = new Float32Array(w * h);
    for (let p = 0; p < n; p++) { const r = ((p < m ? wl : wl + 2) - 1) / 2;
      boxH(a, b, w, h, r); boxV(b, a, w, h, r); }
    return a;
  }
  function boxH(s, d, w, h, r) { const iarr = 1 / (r + r + 1);
    for (let y = 0; y < h; y++) { const o = y * w; let acc = 0;
      for (let j = -r - 1; j < r; j++) acc += s[o + refl(j, w)];
      for (let x = 0; x < w; x++) { acc += s[o + refl(x + r, w)] - s[o + refl(x - r - 1, w)]; d[o + x] = acc * iarr; } } }
  function boxV(s, d, w, h, r) { const iarr = 1 / (r + r + 1);
    for (let x = 0; x < w; x++) { let acc = 0;
      for (let j = -r - 1; j < r; j++) acc += s[refl(j, h) * w + x];
      for (let y = 0; y < h; y++) { acc += s[refl(y + r, h) * w + x] - s[refl(y - r - 1, h) * w + x]; d[y * w + x] = acc * iarr; } } }

  // ---------- colour: sRGB <-> Lab (L 0..255, a/b +128) ----------
  const s2l = c => { c /= 255; return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
  const l2s = c => { c = Math.min(1, Math.max(0, c)); return 255 * (c <= 0.0031308 ? c * 12.92 : 1.055 * Math.pow(c, 1 / 2.4) - 0.055); };
  const LIN = new Float32Array(256 * 16); for (let i = 0; i < LIN.length; i++) LIN[i] = s2l(i / 16);
  const fF = t => t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116;
  const fI = t => t > 0.206893 ? t * t * t : (t - 16 / 116) / 7.787;
  function rgb2lab(r, g, b, o) {
    const R = s2l(r), G = s2l(g), B = s2l(b);
    const X = (0.4124564 * R + 0.3575761 * G + 0.1804375 * B) / 0.95047, Y = 0.2126729 * R + 0.7151522 * G + 0.0721750 * B,
          Z = (0.0193339 * R + 0.1191920 * G + 0.9503041 * B) / 1.08883;
    const fx = fF(X), fy = fF(Y), fz = fF(Z);
    o[0] = (116 * fy - 16) * 2.55; o[1] = 500 * (fx - fy) + 128; o[2] = 200 * (fy - fz) + 128;
  }
  function lab2rgb(L, a, b, o) {
    L = L / 2.55; const fy = (L + 16) / 116, fx = fy + (a - 128) / 500, fz = fy - (b - 128) / 200;
    const X = fI(fx) * 0.95047, Y = fI(fy), Z = fI(fz) * 1.08883;
    o[0] = l2s(3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z);
    o[1] = l2s(-0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z);
    o[2] = l2s(0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z);
  }

  // ---------- geometry helpers ----------
  let Hf = null; // forward homography rect -> still
  function inv3(m) {
    const [a, b, c] = m[0], [d, e, f] = m[1], [g, h, i] = m[2];
    const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g, det = a * A + b * B + c * C;
    return [[A / det, -(b * i - c * h) / det, (b * f - c * e) / det], [B / det, (a * i - c * g) / det, -(a * f - c * d) / det], [C / det, -(a * h - b * g) / det, (a * e - b * d) / det]];
  }
  const clipBox = b => [Math.max(0, Math.floor(b[0])), Math.max(0, Math.floor(b[1])), Math.min(W, Math.ceil(b[2])), Math.min(H, Math.ceil(b[3]))];
  const grow = (b, l, t, r, bt) => [b[0] - l, b[1] - t, b[2] + r, b[3] + (bt === undefined ? t : bt)];
  const union = (...bs) => bs.filter(Boolean).reduce((a, b) => [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]), Math.max(a[3], b[3])]);
  function handGeo(k, ang) {
    const hd = M.hands[k], d = hd.d, p = [-d[1], d[0]], t = ang * Math.PI / 180, u = [Math.sin(t), -Math.cos(t)], n = [-u[1], u[0]];
    return { hd, d, p, u, n, piv: hd.piv, sc: hd.scale, s: spr[k] };
  }
  // bounding box (bbox-local pixels) of a hand sprite drawn at angle ang
  function spriteBox(k, ang) {
    if (!Hf) Hf = inv3(M.Hinv);
    const g = handGeo(k, ang), xs = [], ys = [];
    for (const [sx, sy] of [[0, 0], [g.s.w, 0], [0, g.s.h], [g.s.w, g.s.h]]) {
      const vx = sx - g.piv[0], vy = sy - g.piv[1];
      const du = (vx * g.d[0] + vy * g.d[1]) * g.sc, dn = (vx * g.p[0] + vy * g.p[1]) * g.sc;
      const rx = du * g.u[0] + dn * g.n[0] + 500, ry = du * g.u[1] + dn * g.n[1] + 500;
      const den = Hf[2][0] * rx + Hf[2][1] * ry + Hf[2][2];
      xs.push((Hf[0][0] * rx + Hf[0][1] * ry + Hf[0][2]) / den - X0); ys.push((Hf[1][0] * rx + Hf[1][1] * ry + Hf[1][2]) / den - Y0);
    }
    return [Math.min(...xs) - 3, Math.min(...ys) - 3, Math.max(...xs) + 3, Math.max(...ys) + 3];
  }
  // area of the still that changes when this hand (and its shadow) is drawn
  function changeBox(k, ang) {
    const [ox, oy, bl] = M.shadows[k], r = Math.ceil(4 * bl) + 2, b = spriteBox(k, ang);
    return [b[0] - r, b[1] - r, b[2] + ox + r, b[3] + oy + r];
  }

  // ---------- one hand drawn onto R,G,B (bbox-sized float planes) ----------
  function drawHand(R, G, B, k, ang) {
    const g = handGeo(k, ang), s = g.s, Hi = M.Hinv, [ox, oy, bl] = M.shadows[k];
    const cb = clipBox(changeBox(k, ang)), cw = cb[2] - cb[0], ch = cb[3] - cb[1];
    if (cw <= 0 || ch <= 0) return;
    const A = new Float32Array(cw * ch), C = new Float32Array(cw * ch * 3);
    const sub = [[-0.25, -0.25], [0.25, -0.25], [-0.25, 0.25], [0.25, 0.25]];
    const sb = clipBox(spriteBox(k, ang));
    for (let y = sb[1]; y < sb[3]; y++) for (let x = sb[0]; x < sb[2]; x++) {
      // quick reject using the pixel centre
      { const X = x + X0, Y = y + Y0, den = Hi[2][0] * X + Hi[2][1] * Y + Hi[2][2];
        const rx = (Hi[0][0] * X + Hi[0][1] * Y + Hi[0][2]) / den - 500, ry = (Hi[1][0] * X + Hi[1][1] * Y + Hi[1][2]) / den - 500;
        const du = rx * g.u[0] + ry * g.u[1], dn = rx * g.n[0] + ry * g.n[1];
        const sx = g.piv[0] + (du * g.d[0] + dn * g.p[0]) / g.sc, sy = g.piv[1] + (du * g.d[1] + dn * g.p[1]) / g.sc;
        if (sx < -3 || sy < -3 || sx > s.w + 2 || sy > s.h + 2) continue; }
      let ar = 0, cr = 0, cg = 0, cbv = 0;
      for (const [sx0, sy0] of sub) {
        const X = x + X0 + sx0, Y = y + Y0 + sy0, den = Hi[2][0] * X + Hi[2][1] * Y + Hi[2][2];
        const rx = (Hi[0][0] * X + Hi[0][1] * Y + Hi[0][2]) / den - 500, ry = (Hi[1][0] * X + Hi[1][1] * Y + Hi[1][2]) / den - 500;
        const du = rx * g.u[0] + ry * g.u[1], dn = rx * g.n[0] + ry * g.n[1];
        const sx = g.piv[0] + (du * g.d[0] + dn * g.p[0]) / g.sc, sy = g.piv[1] + (du * g.d[1] + dn * g.p[1]) / g.sc;
        const x0 = Math.floor(sx), y0 = Math.floor(sy);
        if (x0 < -1 || y0 < -1 || x0 >= s.w || y0 >= s.h) continue;
        const fx = sx - x0, fy = sy - y0;
        for (let j = 0; j < 4; j++) {
          const xx = x0 + (j & 1), yy = y0 + (j >> 1);
          if (xx < 0 || yy < 0 || xx >= s.w || yy >= s.h) continue;
          const wgt = ((j & 1) ? fx : 1 - fx) * ((j >> 1) ? fy : 1 - fy), q = (yy * s.w + xx) * 4;
          cr += s.d[q] * wgt; cg += s.d[q + 1] * wgt; cbv += s.d[q + 2] * wgt; ar += s.d[q + 3] * wgt;
        }
      }
      ar /= 4; const a = ar / 255;
      if (a > 0) { const i = (y - cb[1]) * cw + (x - cb[0]); A[i] = a;
        if (a > 1e-4) { const iv = 1 / 4 / a; C[i * 3] = cr * iv; C[i * 3 + 1] = cg * iv; C[i * 3 + 2] = cbv * iv; }
        else { const iv = 1 / 4 / 1e-4; C[i * 3] = cr * iv; C[i * 3 + 1] = cg * iv; C[i * 3 + 2] = cbv * iv; } }
    }
    const sh0 = new Float32Array(cw * ch);
    for (let y = oy; y < ch; y++) for (let x = ox; x < cw; x++) sh0[y * cw + x] = A[(y - oy) * cw + (x - ox)];
    const sh = blur1(sh0, cw, ch, bl);
    for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) {
      const li = y * cw + x, i = (y + cb[1]) * W + x + cb[0], m = 1 - 0.55 * sh[li], a = A[li];
      R[i] = R[i] * m * (1 - a) + C[li * 3] * a; G[i] = G[i] * m * (1 - a) + C[li * 3 + 1] * a; B[i] = B[i] * m * (1 - a) + C[li * 3 + 2] * a;
    }
  }
  function pinBox() { const [px, py, pr] = M.pin; return [px - X0 - pr - 6, py - Y0 - pr - 6, px - X0 + pr + 7, py - Y0 + pr + 7]; }
  function drawPin(R, G, B) {
    const [px, py, pr] = M.pin, b = clipBox(pinBox()), w = b[2] - b[0], h = b[3] - b[1], p0 = new Float32Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const dd = Math.hypot(x + b[0] - (px - X0), y + b[1] - (py - Y0)); p0[y * w + x] = Math.min(1, Math.max(0, pr + 0.5 - dd)); }
    const pin = blur1(p0, w, h, 1.0);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const a = pin[y * w + x], i = (y + b[1]) * W + x + b[0];
      R[i] = R[i] * (1 - a) + base[i * 4] * a; G[i] = G[i] * (1 - a) + base[i * 4 + 1] * a; B[i] = B[i] * (1 - a) + base[i * 4 + 2] * a; }
  }

  // ---------- glass + film look on a region; writes into out (Uint8 RGBA, bbox-sized) ----------
  const FLM = 48; // context margin needed by the blurs (soft 0.9 + bloom 12 reach ~37 px)
  function finish(R, G, B, region, out) {
    const wb = clipBox(region), cb = clipBox(grow(wb, FLM, FLM, FLM)), cw = cb[2] - cb[0], ch = cb[3] - cb[1], N = cw * ch;
    const F = M.FL, lab = [0, 0, 0], rgb = [0, 0, 0], P = [new Float32Array(N), new Float32Array(N), new Float32Array(N)];
    for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) {
      const i = (y + cb[1]) * W + x + cb[0], li = y * cw + x;
      const r = R[i] * gmul[i * 4] / 255 + gadd[i * 4], g = G[i] * gmul[i * 4 + 1] / 255 + gadd[i * 4 + 1], b = B[i] * gmul[i * 4 + 2] / 255 + gadd[i * 4 + 2];
      rgb2lab(r, g, b, lab);
      for (let c = 0; c < 3; c++) lab[c] = lab[c] * 0.4 + ((lab[c] - F.ma[c]) / F.sa[c] * F.sb[c] + F.mb[c]) * 0.6;
      lab2rgb(lab[0], lab[1], lab[2], rgb); P[0][li] = rgb[0]; P[1][li] = rgb[1]; P[2][li] = rgb[2];
    }
    // the blurs reflect at the bbox border (like the reference) but at an inner crop edge the margin makes it irrelevant
    P.forEach((plane, c) => {
      const soft = blur1(plane, cw, ch, 0.9), hi = new Float32Array(N);
      for (let i = 0; i < N; i++) hi[i] = Math.min(255, Math.max(0, soft[i] - 170));
      const bl = boxBlur(hi, cw, ch, 12);
      for (let y = wb[1]; y < wb[3]; y++) for (let x = wb[0]; x < wb[2]; x++) {
        const li = (y - cb[1]) * cw + x - cb[0];
        out[(y * W + x) * 4 + c] = Math.max(0, Math.min(255, Math.round(soft[li] + 0.35 * bl[li])));
      }
    });
  }
  function planes() { const N = W * H, R = new Float32Array(N), G = new Float32Array(N), B = new Float32Array(N);
    for (let i = 0; i < N; i++) { R[i] = base[i * 4]; G[i] = base[i * 4 + 1]; B[i] = base[i * 4 + 2]; } return [R, G, B]; }

  let S0 = null;
  function composeS0() {
    const [R, G, B] = planes(), out = new Uint8Array(W * H * 4).fill(255);
    finish(R, G, B, [0, 0, W, H], out); S0 = out; return out.slice();
  }
  // all second-hand states for one time; only the areas the hands touch are recomputed
  function composeStates(hour, minute, secs) {
    if (!S0) composeS0();
    const [R, G, B] = planes(), aH = ((hour % 12) + minute / 60) * 30, aM = minute * 6;
    drawHand(R, G, B, 'hour', aH); drawHand(R, G, B, 'min', aM);
    const res = {}; let first = null, firstSec = null;
    for (const s of secs) {
      const R2 = R.slice(), G2 = G.slice(), B2 = B.slice();
      drawHand(R2, G2, B2, 'sec', s); drawPin(R2, G2, B2);
      let out, reg;
      if (!first) { out = S0.slice(); reg = union(changeBox('hour', aH), changeBox('min', aM), changeBox('sec', s), pinBox()); }
      else { out = first.slice(); reg = union(changeBox('sec', firstSec), changeBox('sec', s), pinBox()); }
      finish(R2, G2, B2, grow(reg, 44, 44, 44), out);
      if (!first) { first = out; firstSec = s; }
      res[s] = out;
    }
    return res;
  }
  function compose(hour, minute, secAng) { return hour === null ? composeS0() : composeStates(hour, minute, [secAng])[secAng]; }
  return { init, compose, composeStates, get size() { return [W, H]; }, get origin() { return [X0, Y0]; }, get ready() { return ready; } };
})();

// ---------- worker glue ----------
if (typeof WorkerGlobalScope !== 'undefined' && self instanceof WorkerGlobalScope) {
  let latest = null, busy = false;
  async function pump() {
    if (busy || !latest || !Compose.ready) return; busy = true;
    const job = latest; latest = null;
    const t0 = performance.now(), out = Compose.composeStates(job.h, job.m, job.secs);
    self.postMessage({ type: 'done', id: job.id, h: job.h, m: job.m, states: out, ms: performance.now() - t0 }, Object.values(out).map(a => a.buffer));
    busy = false; pump();
  }
  self.onmessage = async e => {
    const d = e.data;
    if (d.type === 'init') {
      try {
        await Compose.init(d.meta, d.dir);
        const t0 = performance.now(), s0 = Compose.compose(null, 0, 0);
        self.postMessage({ type: 'ready', s0, size: Compose.size, ms: performance.now() - t0 }, [s0.buffer]);
        pump();
      } catch (err) { self.postMessage({ type: 'error', msg: String(err) }); }
    } else if (d.type === 'compose') { latest = d; pump(); }
  };
}
