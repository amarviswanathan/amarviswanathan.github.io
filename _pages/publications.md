---
layout: page
permalink: /publications/
title: publications
description: A more updated list can be found at <a href="https://scholar.google.com/citations?user=1YecUQMAAAAJ&hl=en">Google Scholar. </a> 
years: [2009,2013,2016,2017,2018,2020]
nav: true
---

<div class="publications">

{% for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}

</div>
