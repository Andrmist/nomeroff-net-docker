package utils

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt"
)

func AuthHandler(jwtSecret string) gin.HandlerFunc {
	return func(c *gin.Context) {
		authString := strings.Split(c.GetHeader("Authorization"), " ")
		if len(authString) != 2 {
			err := NewHttpError("failed to authorize", "Authorization header is invalid", http.StatusBadRequest)
			c.Error(err)
			c.Abort()
			return
		}
		token, err := jwt.Parse(authString[1], func(t *jwt.Token) (interface{}, error) {
			if _, ok := t.Method.(*jwt.SigningMethodHMAC); !ok {
				return nil, fmt.Errorf("signing method is not valid")
			}
			return []byte(jwtSecret), nil
		})
		if err != nil {
			err := NewHttpError("failed to authorize", "failed to parse authorization token", http.StatusUnprocessableEntity)
			c.Error(err)
			c.Abort()
			return
		}
		if !token.Valid {
			err := NewHttpError("failed to authorize", "authorization token is not valid", http.StatusUnauthorized)
			c.Error(err)
			c.Abort()
			return
		}
	}
}
