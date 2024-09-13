package utils

import (
	"github.com/gin-gonic/gin"
	"go.uber.org/ratelimit"
)

type HTTPLimiter struct {
	ips map[string]ratelimit.Limiter
}

func NewHTTPLimiter() *HTTPLimiter {
	return &HTTPLimiter{
		ips: make(map[string]ratelimit.Limiter),
	}
}

func (h HTTPLimiter) Handler() gin.HandlerFunc {
	return func(c *gin.Context) {
		// cloudflare moment
		ip := c.GetHeader("CF-Connecting-IP")
		limit, ok := h.ips[ip]
		if !ok {
			h.ips[ip] = ratelimit.New(100)
			limit = h.ips[ip]
		}
		limit.Take()
		h.ips[ip] = limit
		c.Next()
	}
}
