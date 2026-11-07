---
layout: page
title: Projects
permalink: /projects/
description: Hands-on practice with machine learning fundamentals.
nav: true
nav_order: 3
---

<div class="projects">
  {% assign featured_projects = site.projects | where: "featured", true | sort: "importance" %}
  <div class="row row-cols-1 row-cols-md-2" style="row-gap: 1.5rem;">
    {% for project in featured_projects %}
      {% include projects.liquid %}
    {% endfor %}
  </div>
</div>
