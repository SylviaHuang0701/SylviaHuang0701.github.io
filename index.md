---
layout: default
title: 首页
---

<div class="home-header">
    <h1 class="home-title">
        <i class="fas fa-water"></i>
        欢迎来到我的技术博客
    </h1>
    <p class="home-subtitle">{{ site.description }}</p>
    <div class="home-actions">
        <a href="/posts" class="btn btn-primary">
            <i class="fas fa-book-open"></i>
            浏览文章
        </a>
        <a href="/about" class="btn btn-secondary">
            <i class="fas fa-info-circle"></i>
            关于博客
        </a>
    </div>
</div>

<div class="posts-section">
    <h2>最新文章</h2>
    <div class="posts-grid">
        {% for post in site.posts limit:6 %}
        <article class="post-card">
            <h2><a href="{{ post.url }}">{{ post.title }}</a></h2>
            <div class="post-meta">
                <time>{{ post.date | date: "%Y年%m月%d日" }}</time>
                {% if post.categories %}
                <span class="categories">{{ post.categories | join: ", " }}</span>
                {% endif %}
            </div>
            <div class="post-excerpt">
                {% if post.excerpt %}
                    {{ post.excerpt | strip_html | truncate: 150 }}
                {% else %}
                    {{ post.content | strip_html | truncate: 150 }}
                {% endif %}
            </div>
            {% if post.tags %}
            <div class="post-tags">
                {% for tag in post.tags %}
                <span class="tag">{{ tag }}</span>
                {% endfor %}
            </div>
            {% endif %}
            <a href="{{ post.url }}" class="read-more">阅读更多 →</a>
        </article>
        {% endfor %}
    </div>
</div>
