"""Assemble the public site: the interactive map plus its link-preview image.

``site/`` is its own git repository (the published GitHub Pages repo), so
this only writes the files it owns and never clears the folder.
"""

import shutil

from ratcast import config


def build_site() -> None:
    site = config.SITE_DIR
    site.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config.MAP_DIR / "rat_map.html", site / "index.html")
    shutil.copyfile(config.FIGURE_DIR / "social_card.png", site / "og-image.png")
    (site / ".nojekyll").touch()  # serve files as-is, no Jekyll processing
    print(f"  wrote site to {site.relative_to(config.PROJECT_ROOT)}/ (index.html, og-image.png)")
