"""
Security middleware for remote access
"""
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.middleware.base import BaseHTTPMiddleware
import secrets
import time
from typing import Dict, List
import ipaddress
import logging

logger = logging.getLogger(__name__)

class SecurityMiddleware(BaseHTTPMiddleware):
    """Basic security middleware for remote access"""
    
    def __init__(self, app, allowed_ips: List[str] = None, rate_limit: int = 100):
        super().__init__(app)
        self.allowed_ips = allowed_ips or []
        self.rate_limit = rate_limit  # requests per minute
        self.rate_tracking: Dict[str, List[float]] = {}
        
    async def dispatch(self, request: Request, call_next):
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        # Check IP whitelist if configured
        if self.allowed_ips and not self._is_ip_allowed(client_ip):
            logger.warning(f"Blocked request from unauthorized IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied from this IP address"
            )
        
        # Rate limiting
        if self._is_rate_limited(client_ip):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Log access
        logger.info(f"Remote access from {client_ip} to {request.url.path}")
        
        response = await call_next(request)
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        # Check for forwarded IP (behind proxy/load balancer)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host
    
    def _is_ip_allowed(self, client_ip: str) -> bool:
        """Check if IP is in whitelist"""
        try:
            client_addr = ipaddress.ip_address(client_ip)
            
            for allowed_ip in self.allowed_ips:
                try:
                    # Handle CIDR notation
                    if '/' in allowed_ip:
                        if client_addr in ipaddress.ip_network(allowed_ip, strict=False):
                            return True
                    else:
                        if client_addr == ipaddress.ip_address(allowed_ip):
                            return True
                except ValueError:
                    continue
            
            return False
            
        except ValueError:
            return False
    
    def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if client has exceeded rate limit"""
        now = time.time()
        minute_ago = now - 60
        
        # Clean old entries
        if client_ip in self.rate_tracking:
            self.rate_tracking[client_ip] = [
                timestamp for timestamp in self.rate_tracking[client_ip]
                if timestamp > minute_ago
            ]
        else:
            self.rate_tracking[client_ip] = []
        
        # Check rate limit
        if len(self.rate_tracking[client_ip]) >= self.rate_limit:
            return True
        
        # Add current request
        self.rate_tracking[client_ip].append(now)
        return False


class SimpleAuth:
    """Simple HTTP Basic authentication"""
    
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.security = HTTPBasic()
    
    def authenticate(self, credentials: HTTPBasicCredentials) -> bool:
        """Verify credentials"""
        is_correct_username = secrets.compare_digest(
            credentials.username, self.username
        )
        is_correct_password = secrets.compare_digest(
            credentials.password, self.password
        )
        
        return is_correct_username and is_correct_password
    
    def get_current_user(self, credentials: HTTPBasicCredentials):
        """Get current authenticated user"""
        if not self.authenticate(credentials):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Basic"},
            )
        return credentials.username


def create_security_config():
    """Create security configuration from environment"""
    import os
    
    config = {
        'enable_auth': os.getenv('ENABLE_AUTH', 'false').lower() == 'true',
        'username': os.getenv('AUTH_USERNAME', 'admin'),
        'password': os.getenv('AUTH_PASSWORD', 'changeme123'),
        'allowed_ips': os.getenv('ALLOWED_IPS', '').split(',') if os.getenv('ALLOWED_IPS') else [],
        'rate_limit': int(os.getenv('RATE_LIMIT', '100'))
    }
    
    return config