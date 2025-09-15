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


# Go后端与内核管理程序controller交互

## `gin.Context`

`gin.Context` 是 `Gin` 框架中的一个核心结构体，它在每个 HTTP 请求的处理过程中被创建，并且贯穿整个请求处理链。它主要包含以下内容：
- 请求数据：包括请求方法、URL、头部信息、请求体等。
- 响应数据：包括响应状态码、响应头部、响应体等。
- 共享数据：用于存储在中间件和处理函数之间共享的数据。
- 请求处理链的控制：提供方法来控制请求的处理流程，如 `Abort`、`Next` 等。
`gin.Context` 的作用类似于一个“请求上下文”，它将 `HTTP` 请求的所有相关信息封装在一起，方便在不同的中间件和处理函数之间传递和共享数据。

## 步骤
### 1. 定义controller接口
``` go
type controller interface{
    AddRule(ctx context.Context, rule *manager.Rule) error
    DeleteRule(ctx context.Context, ruleID uint32) error
	UpdateRule(ctx context.Context, rule *manager.Rule) error
	GetRule(ctx context.Context, ruleID uint32) (*manager.Rule, error)
	ListRules(ctx context.Context) ([]*manager.Rule, error)
	
	// 统计信息接口
	GetGlobalStats(ctx context.Context) (*GlobalStats, error)
	GetRuleStats(ctx context.Context, ruleID uint32) (*RuleStats, error)
	GetTopIPs(ctx context.Context, limit int, sortBy string) ([]*TopIPStats, error)
	
	// 连接管理接口
	GetTCPConnections(ctx context.Context, limit int) ([]*TCPConnection, error)
	GetRuleMatches(ctx context.Context, limit int) ([]*RuleMatch, error)
	
	// 日志接口
	GetLogs(ctx context.Context, level string, limit int) ([]*LogEntry, error)
	
	// 流量趋势接口
	GetTrafficTrend(ctx context.Context, interval string, points int) ([]*TrafficPoint, error)
	
	// 系统控制接口
	ReloadRules(ctx context.Context) error
	GetSystemStatus(ctx context.Context) (*SystemStatus, error)
}
```
### 2. 依赖注入
``` go
   // 集中管理服务所需的各种依赖项
   type ServiceContext struct{
        Config *Config //指向配置信息的指针，可能包含服务运行所需的各种配置参数
        Controller controller.Controller // 添加 controller 接口
   }

 ```

``` go
func newServiceContext(config *Config) *ServiceContext{
    ctrlConfig := &controller.ControllerConfig{
        MapPath:     "/sys/fs/bpf",
        LogLevel:    "INFO",
        BufferSize:  1024,
        Timeout:     30 * time.Second,
    }
    ctrl, err := controller.NewController(ctrlConfig)
    if err != nil {
        log.Printf("Warning: Failed to create controller: %v", err)
        ctrl = nil
    }

    return &ServiceContext{
        Config:     config,
        Controller: ctrl,
    }
}
```
   - 接收一个指向配置信息的指针 config。
   - 创建了一个 controller.ControllerConfig 配置对象 ctrlConfig，设置了内核态 XDP 程序所需的参数（如 BPF 文件系统路径、日志级别、缓冲区大小和超时时间）

### 3. 定义`controller `中间件

位于客户端和服务器的核心业务逻辑之间，用于处理请求和响应。中间件的主要目的是在请求到达最终处理程序之前或之后，执行一些通用的任务或逻辑。

``` go
//典型结构
func middlewareFunc() gin.HandlerFunc{
    return func(c *gin.Context){
        // 在请求处理之前执行的逻辑
		// ...

		// 调用下一个中间件或处理函数
		c.Next()

		// 在请求处理之后执行的逻辑
		// ...
    }
}
```
`controllerMiddleware` 是一个中间件函数，作用是在每个`HTTP `请求的上下文中创建并注入一个 `controller` 实例。这样，后续的处理函数就可以通过 `gin.Context` 访问到这个 `controller` 实例，而无需自己创建。

``` go
func controllerMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		config := &controller.ControllerConfig{
			MapPath:     "/sys/fs/bpf",
			LogLevel:    "INFO",
			BufferSize:  1024,
			Timeout:     30 * time.Second,
		}

		ctrl, err := controller.NewController(config)
		if err != nil {
			log.Printf("Warning: Failed to create controller in middleware: %v", err)
			ctrl = nil
		}
        // 创建controller实例并注入到context
		c.Set("controller", ctrl)
		c.Next()
	}
}
```
## 主要改动



### 2. rule.go 改动

**替换的功能：**
- `getRules()`: 从硬编码数据改为调用`controller.ListRules()`
- `createRule()`: 添加了调用`controller.AddRule()`的逻辑
- `deleteRule()`: 添加了调用`controller.DeleteRule()`的逻辑
- `updateRule()`: 添加了调用`controller.UpdateRule()`的逻辑

**新增的辅助函数：**
- `getController()`: 从gin context获取controller实例
- `getDefaultController()`: 创建默认controller实例
- `generateRuleID()`: 生成规则ID
- `convertServiceRuleToManagerRule()`: 转换service.Rule到manager.Rule
- `convertManagerRuleToServiceRule()`: 转换manager.Rule到service.Rule

**改动内容：**
```go
// 示例：getRules函数改动
func getRules(c *gin.Context) {
	// 从controller获取规则列表
	ctx := c.Request.Context()
	managerRules, err := getController(c).ListRules(ctx)
	if err != nil {
		log.Printf("获取规则列表失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取规则列表失败"})
		return
	}
	// 转换并返回数据
}
```

### 3. status.go 改动

**替换的功能：**
- `getGlobalStatus()`: 从硬编码数据改为调用`controller.GetGlobalStats()`
- `getRuleMatches()`: 从硬编码数据改为调用`controller.GetRuleMatches()`
- `getTopIPs()`: 从硬编码数据改为调用`controller.GetTopIPs()`
- `getTCPConnections()`: 从硬编码数据改为调用`controller.GetTCPConnections()`

**新增的辅助函数：**
- `getController()`: 从gin context获取controller实例
- `getDefaultController()`: 创建默认controller实例

**改动内容：**
```go
// 示例：getGlobalStatus函数改动
func getGlobalStatus(c *gin.Context) {
	// 从controller获取全局统计
	ctx := c.Request.Context()
	globalStats, err := getController(c).GetGlobalStats(ctx)
	if err != nil {
		log.Printf("获取全局统计失败: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "获取全局统计失败"})
		return
	}
	// 转换并返回数据
}
```

## 数据转换

### service.Rule ↔ manager.Rule 转换

**service.Rule → manager.Rule:**
```go
func convertServiceRuleToManagerRule(rule *service.Rule) *manager.Rule {
	managerRule := &manager.Rule{
		RuleID:   rule.ID,
		Protocol: uint8(rule.Protocol),
	}
	
	// 转换动作
	switch rule.Action {
	case "drop":
		managerRule.Action = dao.Drop
	case "accept":
		managerRule.Action = dao.Pass
	}
	
	// 转换IP和端口范围
	// ...
	
	return managerRule
}
```

**manager.Rule → service.Rule:**
```go
func convertManagerRuleToServiceRule(rule *manager.Rule) service.Rule {
	serviceRule := service.Rule{
		ID:        rule.RuleID,
		Protocol:  int(rule.Protocol),
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}
	
	// 转换动作
	switch rule.Action {
	case dao.Pass:
		serviceRule.Action = "accept"
	case dao.Drop:
		serviceRule.Action = "drop"
	}
	
	// 转换IP和端口范围
	// ...
	
	return serviceRule
}
```

## 错误处理

所有controller调用都添加了错误处理：

```go
err := getController(c).AddRule(ctx, managerRule)
if err != nil {
	log.Printf("添加规则到内核失败: %v", err)
	c.JSON(http.StatusInternalServerError, gin.H{"error": "添加规则失败"})
	return
}
```

## 依赖关系

**新增的导入：**
```go
import (
	"context"
	"log"
	"net"
	"github.com/gin-gonic/gin"
	"net/http"
	"strconv"
	"time"

	"github.com/wsm25/XDPTable/controller"
	"github.com/wsm25/XDPTable/model/dao"
	"github.com/wsm25/XDPTable/model/manager"
	"github.com/wsm25/XDPTable/model/service"
)
```

## 配置

Controller配置在多个地方使用：

```go
config := &controller.ControllerConfig{
	MapPath:     "/sys/fs/bpf",
	LogLevel:    "INFO",
	BufferSize:  1024,
	Timeout:     30 * time.Second,
}
```

