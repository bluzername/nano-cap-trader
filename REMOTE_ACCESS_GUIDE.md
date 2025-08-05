# 🌐 Remote Access Guide for NanoCap Trader

This guide shows you how to access your NanoCap Trader system remotely via the internet.

## 🚀 **Quick Start (Recommended)**

### **Method 1: Direct Server Access**

1. **Start the server for remote access:**
```bash
# Simple approach
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Or use the provided script
./start_remote.sh
```

2. **Find your server's IP address:**
```bash
# Public IP
curl -s https://ipinfo.io/ip

# Local network IP
hostname -I | awk '{print $1}'
```

3. **Access remotely:**
- **URL**: `http://YOUR_SERVER_IP:8000`
- **Dashboard**: `http://YOUR_SERVER_IP:8000/dash/`
- **API Docs**: `http://YOUR_SERVER_IP:8000/docs`

---

## 🔒 **Secure Remote Access**

### **Enable Basic Authentication**

Add to your `.env` file:
```env
ENABLE_AUTH=true
AUTH_USERNAME=your_username
AUTH_PASSWORD=your_secure_password_123
```

Now all remote access requires login credentials.

### **IP Whitelisting**

Restrict access to specific IPs:
```env
# Single IP
ALLOWED_IPS=203.0.113.1

# Multiple IPs
ALLOWED_IPS=203.0.113.1,198.51.100.0

# CIDR ranges
ALLOWED_IPS=192.168.1.0/24,10.0.0.0/8
```

### **Rate Limiting**

Control request frequency:
```env
RATE_LIMIT=50  # 50 requests per minute per IP
```

---

## 🔗 **Method 2: ngrok Tunnel (Testing)**

Perfect for quick testing without server configuration:

### **Setup ngrok:**
1. **Install ngrok:**
```bash
# Download and install
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update && sudo apt install ngrok

# Get auth token from https://dashboard.ngrok.com/get-started/your-authtoken
ngrok config add-authtoken YOUR_TOKEN_HERE
```

2. **Start with ngrok:**
```bash
./start_with_ngrok.sh
```

3. **Access via ngrok URL:**
- **URL**: `https://xxxxx.ngrok.io` (shown in terminal)
- **Dashboard**: Check ngrok dashboard at `http://127.0.0.1:4040`

### **ngrok Benefits:**
- ✅ HTTPS encryption automatically
- ✅ No firewall configuration needed
- ✅ Works behind NAT/routers
- ✅ Temporary URLs for testing

---

## 🐳 **Method 3: Docker Deployment**

### **Create Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create non-root user
RUN useradd -m -u 1000 trader && chown -R trader:trader /app
USER trader

EXPOSE 8000

CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

### **Docker Commands:**
```bash
# Build image
docker build -t nano-cap-trader .

# Run container
docker run -d \
  --name nano-cap-trader \
  -p 8000:8000 \
  --env-file .env \
  nano-cap-trader

# View logs
docker logs -f nano-cap-trader
```

---

## ☁️ **Method 4: Cloud Deployment**

### **Render.com (Recommended)**

1. **Connect GitHub repository**
2. **Create new Web Service**
3. **Settings:**
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn main:app -k uvicorn.workers.UvicornWorker -w 1`
   - **Port**: `10000`

4. **Environment Variables:**
   - Add all your API keys from `.env`
   - Set `POLYGON_API_KEY` (required)

5. **Custom Domain:**
   - Free `.onrender.com` subdomain
   - Or connect your own domain

### **Other Cloud Options:**

#### **Railway**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Deploy
railway login
railway init
railway up
```

#### **Heroku**
```bash
# Install Heroku CLI
heroku create nano-cap-trader-YOUR_NAME
git push heroku main
```

#### **DigitalOcean App Platform**
- Connect GitHub repository
- Select Python/Gunicorn runtime
- Set environment variables

---

## 🛡️ **Security Best Practices**

### **Essential Security Settings:**

```env
# Strong authentication
ENABLE_AUTH=true
AUTH_USERNAME=admin_$(date +%s)  # Unique username
AUTH_PASSWORD=$(openssl rand -base64 32)  # Strong password

# IP restrictions (replace with your IP)
ALLOWED_IPS=YOUR_HOME_IP,YOUR_OFFICE_IP

# Conservative rate limiting
RATE_LIMIT=30
```

### **Firewall Configuration:**

```bash
# Ubuntu/Debian firewall
sudo ufw enable
sudo ufw allow 22    # SSH
sudo ufw allow 8000  # NanoCap Trader
sudo ufw status
```

### **HTTPS with Let's Encrypt:**

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

---

## 📊 **Monitoring Remote Access**

### **Check Access Logs:**
```bash
# Real-time logs
tail -f /var/log/nano-cap-trader.log

# Or using Docker
docker logs -f nano-cap-trader
```

### **Monitor Resource Usage:**
```bash
# CPU and memory
htop

# Network connections
netstat -an | grep 8000

# Disk space
df -h
```

---

## 🚨 **Troubleshooting**

### **Common Issues:**

#### **Can't access remotely**
```bash
# Check if service is running
ps aux | grep uvicorn

# Check port binding
netstat -tlnp | grep 8000

# Test local access first
curl http://127.0.0.1:8000/api/health
```

#### **Firewall blocking**
```bash
# Check firewall status
sudo ufw status

# Allow port
sudo ufw allow 8000

# Check iptables
sudo iptables -L
```

#### **Connection refused**
- Ensure server is binding to `0.0.0.0`, not `127.0.0.1`
- Check cloud provider security groups
- Verify port is not already in use

#### **Authentication not working**
```bash
# Test credentials
curl -u username:password http://your-server:8000/api/health

# Check environment variables
env | grep AUTH
```

---

## 📱 **Mobile Access**

The NanoCap Trader interface is mobile-responsive. Access via:

- **Mobile browser**: Same URL as desktop
- **Bookmarks**: Add to home screen for app-like experience
- **PWA**: The interface supports Progressive Web App features

---

## 🎯 **Recommendations by Use Case**

### **Development/Testing**
```bash
# Quick and simple
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### **Personal Use**
```bash
# With basic security
./start_remote.sh
# + Enable AUTH in .env
```

### **Team/Production**
- **Use cloud deployment** (Render.com recommended)
- **Enable HTTPS**
- **Set up monitoring**
- **Regular backups**

### **Demo/Presentation**
```bash
# Use ngrok for temporary access
./start_with_ngrok.sh
```

---

## 🔧 **Advanced Configuration**

### **Custom Domain Setup:**

1. **Point domain to server IP**
2. **Configure reverse proxy (nginx):**

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

3. **Add SSL certificate**
4. **Update DNS records**

---

## 📞 **Support**

If you encounter issues:

1. **Check logs** for error messages
2. **Verify environment variables** are set correctly
3. **Test local access** first (`http://127.0.0.1:8000`)
4. **Check network connectivity** and firewall settings
5. **Review security settings** if authentication fails

**Ready to go remote? Start with:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then access via: `http://YOUR_SERVER_IP:8000` 🚀