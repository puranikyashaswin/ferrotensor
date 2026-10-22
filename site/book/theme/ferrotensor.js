// Link from the guide's menu bar back to the landing page one level up.
(function () {
  "use strict";
  var buttons = document.querySelector(".right-buttons");
  if (!buttons || typeof path_to_root === "undefined") return;
  var link = document.createElement("a");
  link.className = "ft-home";
  link.href = path_to_root + "../";
  link.textContent = "ferrotensor home";
  buttons.insertBefore(link, buttons.firstChild);
})();
