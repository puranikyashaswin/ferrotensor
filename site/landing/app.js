// Install tabs and copy buttons. Without this script every command stays visible.
(function () {
  "use strict";

  document.querySelectorAll("[data-tabs]").forEach(function (root) {
    var tabs = Array.prototype.slice.call(root.querySelectorAll('[role="tab"]'));

    function select(tab, focus) {
      tabs.forEach(function (t) {
        var on = t === tab;
        t.setAttribute("aria-selected", on ? "true" : "false");
        t.tabIndex = on ? 0 : -1;
        document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
      });
      if (focus) tab.focus();
    }

    tabs.forEach(function (tab, i) {
      tab.addEventListener("click", function () { select(tab, false); });
      tab.addEventListener("keydown", function (e) {
        var next = null;
        if (e.key === "ArrowRight") next = tabs[(i + 1) % tabs.length];
        else if (e.key === "ArrowLeft") next = tabs[(i - 1 + tabs.length) % tabs.length];
        else if (e.key === "Home") next = tabs[0];
        else if (e.key === "End") next = tabs[tabs.length - 1];
        if (next) {
          e.preventDefault();
          select(next, true);
        }
      });
    });

    root.classList.add("is-live");
    select(tabs[0], false);
  });

  function fallbackCopy(text) {
    var area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    var ok = false;
    try { ok = document.execCommand("copy"); } catch (err) { ok = false; }
    document.body.removeChild(area);
    return ok ? Promise.resolve() : Promise.reject(new Error("copy failed"));
  }

  document.querySelectorAll("[data-copy]").forEach(function (btn) {
    var label = btn.textContent;
    var timer = 0;
    btn.addEventListener("click", function () {
      var text = btn.getAttribute("data-copy");
      var done = navigator.clipboard && window.isSecureContext
        ? navigator.clipboard.writeText(text).catch(function () { return fallbackCopy(text); })
        : fallbackCopy(text);
      done.then(
        function () {
          btn.textContent = "Copied";
          btn.setAttribute("data-state", "done");
          var status = document.getElementById("copy-status");
          if (status) status.textContent = "Copied: " + text;
        },
        function () { btn.textContent = "Press Ctrl+C"; }
      );
      clearTimeout(timer);
      timer = setTimeout(function () {
        btn.textContent = label;
        btn.removeAttribute("data-state");
      }, 1800);
    });
  });
})();
