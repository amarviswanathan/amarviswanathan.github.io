---
layout: page
title: Publications by year
description:
sorting: year
link_to: type
yearstart: 2002
---

{% include bib_intro.html %}
{% assign year_end = "now" | date: "%Y" %}

{% for y in (page.yearstart..year_end) reversed %}
  <h3 class="year">{{y}}</h3>
  {% bibliography -f {{site.scholar.bibliography}} -q @*[year={{y}}]* %}
{% endfor %}