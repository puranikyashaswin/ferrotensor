// How-to guide: show the hub or one guide by URL hash, filter and search
// the topic list. Without this script every guide is shown in order.
(function () {
  "use strict";

  var main = document.querySelector(".gmain");
  var guides = Array.prototype.slice.call(document.querySelectorAll(".gmain > .g"));
  var sideLinks = Array.prototype.slice.call(document.querySelectorAll(".gside a"));
  var rows = Array.prototype.slice.call(document.querySelectorAll(".acc li"));
  var accs = Array.prototype.slice.call(document.querySelectorAll(".acc"));
  var chips = Array.prototype.slice.call(document.querySelectorAll(".chip"));
  var search = document.getElementById("gsearch");
  var noHits = document.querySelector(".no-hits");
  var filter = "all";

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
      if (on) {
        var sec = a.closest("details");
        if (sec) sec.open = true;
      }
    });
    if (current) {
      document.title = current.querySelector("h2").textContent + " | Ferrotensor how-to guide";
      window.scrollTo(0, 0);
    } else {
      document.title = "How-to guide: Ferrotensor step by step";
    }
  }

  function apply() {
    var q = (search.value || "").trim().toLowerCase();
    var any = false;
    rows.forEach(function (li) {
      var tagOk = filter === "all" || li.getAttribute("data-tags").split(" ").indexOf(filter) >= 0;
      var textOk = !q || li.getAttribute("data-text").indexOf(q) >= 0;
      li.hidden = !(tagOk && textOk);
      if (!li.hidden) any = true;
    });
    accs.forEach(function (acc) {
      var visible = acc.querySelectorAll("li:not([hidden])").length;
      acc.hidden = visible === 0;
      if (q || filter !== "all") acc.open = visible > 0;
    });
    noHits.hidden = any;
  }

  chips.forEach(function (chip) {
    chip.addEventListener("click", function () {
      filter = chip.getAttribute("data-filter");
      chips.forEach(function (c) { c.setAttribute("aria-pressed", c === chip ? "true" : "false"); });
      apply();
    });
  });

  search.addEventListener("input", function () {
    if (main.classList.contains("one")) {
      history.pushState(null, "", "#start");
      show();
    }
    apply();
  });
  search.addEventListener("keydown", function (e) {
    if (e.key === "Enter") {
      var first = document.querySelector(".acc li:not([hidden]) a");
      if (first) location.hash = first.getAttribute("href");
    } else if (e.key === "Escape") {
      search.value = "";
      apply();
    }
  });
  document.addEventListener("keydown", function (e) {
    var t = e.target;
    if (e.key === "/" && t !== search && !/^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) {
      e.preventDefault();
      search.focus();
    }
  });

  window.addEventListener("hashchange", show);
  show();
  apply();
})();
