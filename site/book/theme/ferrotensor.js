// Guide extras: a link back to the landing page, and the sidebar toggle as a
// real button (seo.py turns mdBook's <label> into a <button>, so it no longer
// flips the checkbox on its own). Loaded after book.js, so mdBook's own click
// handler has already run when this one flips the checkbox.
(function () {
  "use strict";
  var buttons = document.querySelector(".right-buttons");
  if (buttons && typeof path_to_root !== "undefined") {
    var link = document.createElement("a");
    link.className = "ft-home";
    link.href = path_to_root + "../";
    link.textContent = "ferrotensor home";
    buttons.insertBefore(link, buttons.firstChild);
  }

  var toggle = document.getElementById("mdbook-sidebar-toggle");
  var box = document.getElementById("mdbook-sidebar-toggle-anchor");
  if (toggle && box && toggle.tagName === "BUTTON") {
    toggle.addEventListener("click", function () {
      box.checked = !box.checked;
      box.dispatchEvent(new Event("change"));
    });
  }
})();
