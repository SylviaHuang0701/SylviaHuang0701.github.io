---
title: "解决 VMware Workstation Network Unreachable"
date: 2025-06-28
categories: [虚拟化]
tags: [vmware, network, ubuntu, dhclient]
---

## 问题描述
在 VMware Workstation 中启动 Ubuntu 虚拟机后，发现网络连接异常，出现 `network unreachable` 错误。尝试重启 NetworkManager 服务无效：

```bash
sudo systemctl restart NetworkManager
```

1. 检查 DHCP 客户端进程：
```bash
ps aux | grep dhclient
```
输出显示没有活动的 dhclient 进程：
```
3713  0.0  0.0   9040   720 pts/0    S+   03:43   0:00 grep --color=auto dhclient
```

2. 手动启动 DHCP 客户端获取 IP 地址：

```bash
sudo dhclient ens33  # ens33 是默认网络接口名
```

执行后立即生效，网络连接恢复正常。

## 原因分析
这种情况通常发生在：
- 虚拟机从休眠状态恢复时
- 主机网络配置变更后
- VMware 虚拟网络服务未正确初始化

DHCP 客户端进程（dhclient）未能自动启动，导致系统无法获取 IP 地址。
