---
layout: default
title: 标签
permalink: /tags/
---

<div class="tags-page">
    <header class="page-header">
        <h1 class="page-title">
            <i class="fas fa-tags"></i>
            标签云
        </h1>
        <p class="page-description">按标签浏览文章</p>
    </header>

    <div class="tags-content">
        {% assign tags = site.tags | sort %}
        {% for tag in tags %}
        <div class="tag-section">
            <h2 class="tag-name" id="{{ tag[0] | slugify }}">
                <i class="fas fa-tag"></i>
                {{ tag[0] }}
                <span class="tag-count">({{ tag[1].size }})</span>
            </h2>
            <div class="tag-posts">
                {% for post in tag[1] %}
                <article class="tag-post">
                    <time class="post-date">{{ post.date | date: "%Y-%m-%d" }}</time>
                    <a href="{{ post.url }}" class="post-title">{{ post.title }}</a>
                </article>
                {% endfor %}
            </div>
        </div>
        {% endfor %}
    </div>
</div> 