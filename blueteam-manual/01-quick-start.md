# Часть 1 — Quick Start (шпаргалка на 1–2 страницы)

Держать открытой всегда. Подробности: [02-playbook.md](02-playbook.md) (анализ и атаки), [03-reference.md](03-reference.md) (конфигурации), [05-troubleshooting.md](05-troubleshooting.md) (откат).

---

## 1.1 Первые 15–30 минут на КАЖДОМ активе

Выполнять по порядку. Все команды read-only, кроме помеченных `[ЗАПИСЬ]` (создание бэкапов — это не изменение защищаемых приложений, но проверь правила).

```bash
# --- ШАГ 0. Определиться с доступом и ОС -----------------------------------
id                                   # кто я; uid=0 = root, иначе нужен sudo
sudo -n true 2>/dev/null && echo "sudo OK без пароля" || echo "sudo спросит пароль"
cat /etc/os-release | head -2        # Debian/Ubuntu vs RHEL — разные пути логов
hostname -I                          # мои IP (записать в logbook — НИКОГДА не блокировать)
timedatectl status | grep -E "synchronized|Time zone"   # время важно для корреляции
```

Интерпретация: если `sudo` нет — сразу идти к организаторам; без прав многое заблокировано. Время не синхронизировано → корреляция логов между хостами будет врать (отметить в logbook, учесть сдвиг).

```bash
# --- ШАГ 1. Карта активов: что это за машина и что крутит ------------------
ss -tulpn                            # [root] ВСЕ слушающие порты + процессы
systemctl list-units --type=service --state=running
ps auxf | head -60                   # дерево процессов: веб? БД? очереди?
docker ps -a 2>/dev/null             # [если Docker] что в контейнерах
```

Интерпретация: колонка `Local Address:Port` = точки входа (например `0.0.0.0:80`, `0.0.0.0:3000`).
Имя процесса в колонке справа = владельцу порта. Порты на `127.0.0.1` — внутренние (БД, админки), на `0.0.0.0`/`::` — внешние (целевые для атак).

```bash
# --- ШАГ 2. Назначение и админ-интерфейсы ----------------------------------
ls -la /var/www 2>/dev/null; ls -la /opt 2>/dev/null; ls -la /srv 2>/dev/null
sudo nginx -T 2>/dev/null | grep -E "server_name|listen|root|proxy_pass" | head -30
sudo apachectl -S 2>/dev/null
ss -tlnp | awk '$4 !~ /127\.0\.0\.1|::1/'    # только внешне-доступные сервисы
```

Интерпретация: `server_name`+`listen` = какие сайты живут на хосте; `proxy_pass` = какой backend за прокси (его тоже защищаем, но нельзя ронять). Внешне-доступные админ-панели — кандидат №1 на прикрытие (см. [03-reference.md](03-reference.md) H.1).

```bash
# --- ШАГ 3. Фиксация исходного состояния (baseline) ------------------------
mkdir -p /root/bluelab && cd /root/bluelab
sudo ss -tulpn  > net-ports.txt
ps auxf         > processes.txt
df -h           > disk.txt; free -h > mem.txt; top -bn1 | head -15 > top.txt
uptime          > load.txt
curl -s -o /dev/null -w "%{http_code} %{time_total}\n" http://127.0.0.1/ > health-baseline.txt 2>&1   # порт подставить свой
```

Это «до». То же самое снимаешь после каждого крупного изменения — разница = доказательство, что защита не сломала сервис (см. метрики в [04-reporting.md](04-reporting.md)).

```bash
# --- ШАГ 4. Бэкапы конфигураций [ЗАПИСЬ] ------------------------------------
sudo tar -czf /root/bluelab/etc-backup-$(date +%F-%H%M).tar.gz \
     /etc/nginx /etc/apache2 /etc/fail2ban /etc/suricata /etc/modsecurity \
     /etc/nftables.conf /etc/audit /etc/ssh /etc/hosts 2>/dev/null
sudo nft list ruleset  > /root/bluelab/nft-baseline.txt 2>/dev/null
sudo iptables-save     > /root/bluelab/iptables-baseline.txt 2>/dev/null
sudo crontab -l        > /root/bluelab/cron-root.txt 2>/dev/null
# Скопировать архив с хоста (если есть второй хост/ноутбук с доступом):
# scp /root/bluelab/etc-backup-*.tar.gz user@my-laptop:/tmp/
```

Правило: любой сброс окружения теперь не страшен, любой откат возможен (процедуры — [05-troubleshooting.md](05-troubleshooting.md)).

```bash
# --- ШАГ 5. Логи: где что лежит --------------------------------------------
ls -la /var/log/ | head -30
ls -la /var/log/nginx /var/log/apache2 /var/log/httpd /var/log/mysql 2>/dev/null
ls -la /var/log/auth.log /var/log/secure 2>/dev/null     # Debian | RHEL
sudo journalctl --list-boots | tail -3
```

```bash
# --- ШАГ 6. Уже установленные защитные средства ----------------------------
for b in nginx apache2 suricata zeek fail2ban-client nft iptables auditctl \
         crowdsec docker podman jq curl dig; do command -v $b >/dev/null && echo "есть: $b"; done
ls -d /var/ossec 2>/dev/null && echo "есть: Wazuh agent/manager"
systemctl is-active nftables firewalld ufw fail2ban suricata 2>/dev/null
```

```bash
# --- ШАГ 7. Первая проверка на уже идущие атаки (5 минут) -------------------
sudo tail -100 /var/log/auth.log 2>/dev/null | grep -cE "Failed|Invalid" || sudo journalctl -u ssh -S "1 hour ago" | grep -cE "Failed|Invalid"
sudo awk '{print $1}' /var/log/nginx/access.log 2>/dev/null | sort | uniq -c | sort -rn | head -10
```

Интерпретация: сотни Failed password = brute force уже идёт. Один IP с тысячей запросов = скан/флуд. Подробнее: [02-playbook.md](02-playbook.md) раздел 2.2.

## 1.2 Скрипт обследования `recon.sh` (read-only)

Скопировать на актив, запустить `sudo bash recon.sh > recon-$(hostname).txt`. Разослать результаты себе — это каркас отчёта.

```bash
#!/usr/bin/env bash
# recon.sh — только чтение. Проверено: Ubuntu 22.04, Debian 12 (root/sudo).
set +e
echo "=== HOST $(hostname) $(date -u +%FT%TZ) ==="
echo "--- OS / kernel"; cat /etc/os-release | head -2; uname -a; uptime
echo "--- MEMORY / DISK"; free -h; df -h --exclude=tmpfs --exclude=devtmpfs
echo "--- INODES (переполнение = отказ сервиса)"; df -i | tail -n +2 | awk '$5+0>80'
echo "--- LISTENING PORTS"; ss -tulpn
echo "--- ESTABLISHED (топ удалённых IP)"; ss -tn state established | awk 'NR>1 {print $5}' | rev | cut -d: -f2- | rev | sort | uniq -c | sort -rn | head -15
echo "--- TOP CPU"; ps aux --sort=-%cpu | head -8
echo "--- TOP MEM"; ps aux --sort=-%mem | head -8
echo "--- RUNNING SERVICES"; systemctl list-units --type=service --state=running --no-pager --no-legend | awk '{print $1}'
echo "--- ENABLED (автостарт)"; systemctl list-unit-files --state=enabled --no-pager --no-legend | awk '{print $1}' | head -40
echo "--- USERS с shell"; awk -F: '$7 !~ /nologin|false/ {print $1, $3, $7}' /etc/passwd
echo "--- ROOT-подобные аккаунты"; awk -F: '$3 == 0 {print $1}' /etc/passwd
echo "--- SUDO группа"; getent group sudo wheel 2>/dev/null
echo "--- LAST LOGINS"; last -a | head -15
echo "--- FAILED LOGINS (счётчик)"; lastb 2>/dev/null | head -10; lastb 2>/dev/null | wc -l
echo "--- CRON"; ls -la /etc/cron* 2>/dev/null | head -20; crontab -l 2>/dev/null
echo "--- DOCKER"; docker ps -a --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null
echo "--- FIREWALL STATE"; systemctl is-active nftables firewalld ufw 2>/dev/null; nft list ruleset 2>/dev/null | head -50; iptables -S 2>/dev/null | head -30
echo "--- SEC TOOLS"; for b in suricata zeek fail2ban-client auditctl crowdsec nginx; do command -v $b; done; ls -d /var/ossec 2>/dev/null
echo "--- SECURITY-RELATED JOURNAL (последние ошибки)"; journalctl -p warning -S "1 hour ago" --no-pager | tail -20
echo "=== END ==="
```

## 1.3 Экстренные карточки (если горит)

**Сервис стал недоступен** (подробный алгоритм — [02-playbook.md](02-playbook.md) 2.4.1):
```bash
systemctl status <service> --no-pager -l        # упал? restarted?
ss -tlnp | grep <порт>                          # слушает ли порт
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:<порт>/   # отвечает ли локально
sudo journalctl -u <service> -S "15 minutes ago" --no-pager | tail -30   # причина
# Если виновато последнее изменение конфига — откатить бэкап (05-troubleshooting.md)
```

**WAF ломает сайт (всплеск 403):**
```bash
sudo systemctl cat nginx | grep -i modsec       # понять, где включен ModSecurity
# Быстро: вернуть DetectionOnly (только пишет в лог, не блокирует)
# в файле modsecurity.conf: SecRuleEngine DetectionOnly
sudo nginx -t && sudo systemctl reload nginx    # откат: вернуть On
```

**Забанили самого себя / легитимного:**
```bash
sudo fail2ban-client set sshd unbanip <IP>      # fail2ban
sudo nft delete element inet bluelab blocklist { <IP> }   # nftables bluelab
sudo iptables -D INPUT -s <IP> -j DROP          # iptables (посмотреть: iptables -S | grep <IP>)
```

**Закончилось место на диске:**
```bash
df -h | awk '$5+0>85'                            # где переполнение
sudo du -xh --max-depth=2 /var 2>/dev/null | sort -h | tail -10
# СНАЧАЛА заархивировать логи как доказательства (04-reporting.md), потом:
sudo journalctl --vacuum-size=300M               # сжать старые журналы (безопасно)
```

**Подозрительный IP — что делать (минимальное действие):**
```bash
sudo grep -F "<IP>" /var/log/nginx/access.log | tail -20       # что он делает
sudo grep -F "<IP>" /var/log/auth.log 2>/dev/null | tail       # пытался ли войти
# Уверен, что вредонос И подозреваемых мало → временный бан 15 минут:
sudo nft add element inet bluelab blocklist { <IP> timeout 15m }   # 03-reference.md H.6
```

## 1.4 Симптом → первая команда

| Симптом | Первая команда | Куда дальше |
|---|---|---|
| Веб отдаёт 500/502 | `sudo tail -20 /var/log/nginx/error.log` | 02-playbook 2.4.1, сценарий G.4 |
| Сотни 404 подряд | `sudo awk '$9==404 {print $1}' access.log \| sort \| uniq -c \| sort -rn \| head` | 2.2 (file enumeration) |
| 403 всплеск | `sudo grep -oE 'id "[0-9]+"' /var/log/modsec_audit.log \| sort \| uniq -c \| sort -rn \| head` | WAF FP — 05-troubleshooting 5.4 |
| Много SSH Failed password | `sudo lastb \| awk '{print $3}' \| sort \| uniq -c \| sort -rn \| head` | 2.2 (brute force) |
| Растёт load average | `top -bn1`, `ps aux --sort=-%cpu \| head` | 2.1.3 |
| Диск 90%+ | `df -h`, `du -xh --max-depth=2 /var` | карточка выше, G.8 |
| Wazuh-алерт уровень 10+ | открыть алерт, найти в логах совпадение по времени | 2.5 (decision trees) |
| Suricata alert | `sudo tail -50 /var/log/suricata/fast.log` | 2.3.5 |

## 1.5 Один экран команд «на каждый час»

```bash
# Раз в час: дежурный обзор (все read-only)
sudo ss -tn state established | awk 'NR>1 {print $5}' | rev | cut -d: -f2- | rev | sort | uniq -c | sort -rn | head -5   # аномальные соседи
sudo awk '{print $9}' /var/log/nginx/access.log | sort | uniq -c | sort -rn | head -5                                    # профиль кодов
sudo awk '{print $4}' /var/log/nginx/access.log | cut -d: -f1-3 | uniq -c | tail -10                                     # всплески RPS по минутам
sudo tail -5 /var/log/fail2ban.log 2>/dev/null
sudo tail -5 /var/log/suricata/fast.log 2>/dev/null
sudo journalctl -p err -S "30 minutes ago" --no-pager | tail -10
curl -s -o /dev/null -w "%{http_code} %{time_total}s\n" http://127.0.0.1/     # жив ли сервис
```
