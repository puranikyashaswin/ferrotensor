// How-to guide: show the overview or one guide by URL hash, search the guide
// list, and copy commands. Without this script every guide is shown in order.
(function () {
  "use strict";

  var main = document.querySelector(".gmain");
  var guides = Array.prototype.slice.call(document.querySelectorAll(".gmain > .g"));
  var sideLinks = Array.prototype.slice.call(document.querySelectorAll(".gside nav a"));
  var rows = Array.prototype.slice.call(document.querySelectorAll(".isec li"));
  var secs = Array.prototype.slice.call(document.querySelectorAll(".isec"));
  var search = document.getElementById("gsearch");
  var noHits = document.querySelector(".no-hits");
  var status = document.getElementById("copy-status");

  function show() {
    var id = decodeURIComponent(location.hash.slice(1));
    var current = null;
    guides.forEach(function (g) {
      var on = g.id === id;
      g.classList.toggle("on", on);
      if (on) current = g;
    });
    main.classList.toggle("one", !!current);
    sideLinks.forEach(function (a) {
      var target = a.getAttribute("href").slice(1);
      var on = current ? target === current.id : target === "start";
      if (on) a.setAttribute("aria-current", "true"); else a.removeAttribute("aria-current");
    });
    document.title = current
      ? current.querySelector("h2").textContent + " | Ferrotensor how-to guide"
      : "How-to guide: Ferrotensor step by step";
    window.scrollTo(0, 0);
  }

  function filter() {
    var q = search.value.trim().toLowerCase();
    var any = false;
    rows.forEach(function (li) {
      li.hidden = !!q && li.getAttribute("data-text").indexOf(q) < 0;
      if (!li.hidden) any = true;
    });
    secs.forEach(function (s) { s.hidden = s.querySelectorAll("li:not([hidden])").length === 0; });
    sideLinks.forEach(function (a) {
      var t = a.getAttribute("data-text");
      if (t) a.parentNode.hidden = !!q && t.indexOf(q) < 0;
    });
    noHits.hidden = any;
  }

  search.addEventListener("input", function () {
    if (main.classList.contains("one")) {
      history.pushState(null, "", "#start");
      show();
    }
    filter();
  });
  search.addEventListener("keydown", function (e) {
    if (e.key === "Enter") {
      var first = document.querySelector(".isec li:not([hidden]) a");
      if (first) location.hash = first.getAttribute("href");
    } else if (e.key === "Escape") {
      search.value = "";
      filter();
    }
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "/" && !/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) {
      e.preventDefault();
      search.focus();
    }
  });

  document.querySelectorAll(".gcopy").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var text = btn.getAttribute("data-copy");
      var done = function () {
        btn.textContent = "Copied";
        btn.setAttribute("data-state", "done");
        if (status) status.textContent = "Copied to the clipboard";
        setTimeout(function () { btn.textContent = "Copy"; btn.removeAttribute("data-state"); }, 1600);
      };
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(done, function () {});
      }
    });
  });

  window.addEventListener("hashchange", show);
  if (location.hash) show(); else { main.classList.remove("one"); show(); }
  filter();
})();
