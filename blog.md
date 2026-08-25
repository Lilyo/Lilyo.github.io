---
layout: singlePage
title: "Blog Posts"
---

# Blog Posts

<ul class="post-list">
  {% for post in site.posts %}
    {% unless post.draft %}
    <li>
      <a class="post-list-title" href="{{ post.url }}">{{ post.title }}</a>
      <span class="date-badge">{{ post.date | date: "%B %e, %Y" }}</span>
    </li>
    {% endunless %}
  {% endfor %}
</ul>
