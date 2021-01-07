---
layout: page
permalink: /publications/
title: publications
description: Publications in reverse chronological order
years: [2009,2013,2016,2017,2018,2020]
nav: true
---

<div class="publications">

{% for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}

</div>
