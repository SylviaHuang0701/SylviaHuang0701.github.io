---
title: "gin框架学习记录"
date: 2025-06-29
categories: [前后端]
tags: [Gin]
---
# gin框架学习记录
## 最基础实例
```go
package main

import (
	"github.com/gin-gonic/gin"
)

func main() {
	Server := gin.Default() //创建服务

	//请求相应、处理
	Server.GET("/hello", func(context *gin.Context) {
		context.JSON(200, gin.H{"msg": "hello, world"})
	})
	//端口
	//相应页面给前端
	Server.Run(":8082")
}

```
- "/hello"是路径，相当于当用户访问“https：//localhost:8082/hello”时，用相应的处理函数func进行响应
- context 对象是gin框架进行http相应的核心之一，一个http请求就创建一个context对象.包括了获取的请求信息（如json请求体）、相应的信息（如相应的json）、处理流程控制（如中断、返回错误）
- context.JSON返回JSON的格式相应（状态码200,和一个gin中的类型）
- gin.H是一个map（字典）msg是key,"hello,world"是这个条目的内容。使用context.JSON作用，会将上面的gin.H转化为json,用于请求相应
## 响应相关常用函数
- 用json响应（如上）
	```go
		Server.GET("/hello", func(context *gin.Context) {
		context.JSON(200, gin.H{"msg": "hello, world"})
	})
	```
- 用html响应(返回一个动态信息)
	```go
	Server.GET("/index", func(context *gin.Context){
		context.HTML(http.StatusOK, "index.html",gin.H{
			"msg": "hello 1234"
		})
	})
	```
	此时，在前端只需要使用{{.msg}}，就能显示msg的内容,即实现了从后端->前端信息
## 加载页面相关
- 静态页面法1
用Static方法，指定网页路径、文件位置
```go
	//提供静态文件服务
	router.Static("/web", "./htmlfiles") // 网页路径为localhost/web, 文件在htmlfiles文件夹下
```
- 静态方法2
```go
	router.LoadHTMLGlob("htmlfiles/*")
```
指加载相应文件夹下所有的html文件
## 获取请求中参数
- 传统url请求
	对于形如usl?userid=xxx&username=abc这样用？拼接的参数
	```go
	ginServer.GET("/user/info", func(context *gin.Context){
		userid := context.Query("userid") //Query函数用于查询发送过来的url请求
		context.JSON(http.StatusOK, gin.H{"userid": userid})
	})
	```
	当输入host/user/info?userid=0&username=aaa，就能够返回userid的json文件
- restful api请求
	restful api的请求长得像这样：host/user/info/11/abc(后面是参数)
	```go
	ginServer.GET("/user/info/:userid/:username", func(context *gin.Context){
		userid := context.Param("userid") //Param函数用于从restful
		context.JSON(http.StatusOK, gin.H{"userid": userid})
	})	
	```
- 接收并解析前端给后端发来的json文件
	```go
	Server.POST("/json", func(context *gin.Context){
		data,error := context.GetRawData()
		//错误处理
		var m map[string]interface{}
		_ = json.Unmarshal(data, &m) //这个函数会把收到的data(实际上是一个slice, 解析到map里,变成字典，就和可以返回的json文件一样
		context.JSON(http.StatusOK, m)
	})
	```
- 获取从前端发来的表单（<form ...>）
	用`context.PostForm("key")`

## 路由重定向
```go
ginServer.GET("/test", func(context *gin.Context){
	context.Redirect(301, "new url") //301表示重定义的code,也可以用http.StatusMovedPermanently
})
```
## 路由组
把很多路由组合到一起，这个很有用
```go
userGroup ：= ginServer.Group("/user"){
	userGroup.GET("/add",func(...))
	userGroup.GET("/login",func(...))
	//。。。。。。
}
```
## 中间拦截
比如说在处理前需要中间登陆验证
先定义一个中间处理函数
```go
func midHandler() (gin.HandlerFunc) {
	return func(context *gin.Context){
		context.Set("usersession", "111") //可以给context设一个条目，key为usersession,之后的处理时就可以读到
		context.Next() //放行，可以允许推出中间处理
		//XXX
	}
}
```
在正式处理时，在本来的`ginServer.GET("/a", func(context *gin.Context)`的中间加上处理函数，
如`ginServer.GET("/test", midHandler(), func(context *gin.Context)`

## CORS跨域资源共享机制

``` go
func corsMiddleware() gin.HanlderFunc{
	c.Header("Access-Control-Allow-Origin","*") //设置允许访问该资源的域名
	c.Header("Access-Control-Allow-Methods","GET,POST,PUT,DELETE,OPTIONS") //设置允许的HTTP方法
	c.Header("Access-Control-Allow-Headers", "Origin, Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization") //设置允许的请求头

	// 处理预检请求
	if c.Requested.Method == "OPTIONS"{
		c.AbortWithStatus(204) //返回204 no content
		return
	}
	c.Next() //处理后续请求
}
```

### 1. 简单请求（Simple Request）

当请求满足以下条件时，浏览器会直接发送请求：
- 使用 GET、HEAD 或 POST 方法
- 请求头仅包含：Accept、Accept-Language、Content-Language、Content-Type
- Content-Type 为：`application/x-www-form-urlencoded`, `multipart/form-data` 或 `text/plain`

**处理流程：**
1. 浏览器发送带 `Origin` 头的实际请求
2. 服务器响应包含 `Access-Control-Allow-Origin` 头
3. 浏览器检查响应头，允许或拒绝访问

### 2. 预检请求（Preflight Request）

对于不符合简单请求条件的请求，浏览器会先发送 OPTIONS 预检请求。

**预检请求流程：**
``` mermaid
sequenceDiagram
    participant Browser
    participant Server
    
    Browser->>Server: OPTIONS 请求
    Note over Browser: 包含:<br/>Origin<br/>Access-Control-Request-Method<br/>Access-Control-Request-Headers
    Server->>Browser: 204 响应
    Note over Server: 包含:<br/>Access-Control-Allow-Origin<br/>Access-Control-Allow-Methods<br/>Access-Control-Allow-Headers
    Browser->>Server: 实际请求 (GET/POST/PUT/DELETE)
    Server->>Browser: 实际响应
```

### 请求头（由浏览器自动添加）

| 请求头 | 说明 |
|--------|------|
| `Origin` | 发起请求的源（协议+域名+端口） |
| `Access-Control-Request-Method` | 预检请求中声明实际请求的方法 |
| `Access-Control-Request-Headers` | 预检请求中声明实际请求的头部 |

### 响应头（由服务器设置）

| 响应头 | 说明 | 示例值 |
|--------|------|--------|
| `Access-Control-Allow-Origin` | 允许访问的源 | `*` 或 `https://example.com` |
| `Access-Control-Allow-Methods` | 允许的HTTP方法 | `GET, POST, PUT, DELETE` |
| `Access-Control-Allow-Headers` | 允许的请求头 | `Content-Type, Authorization` |
| `Access-Control-Allow-Credentials` | 是否允许发送凭证（如cookies） | `true` |
| `Access-Control-Max-Age` | 预检请求缓存时间（秒） | `86400` |
