package config

import (
	"errors"
	"strings"

	"github.com/minio/minio-go/v7"
	"github.com/sirupsen/logrus"
)

type config struct {
	LogParams struct {
		Level string `json:"level"`
		Dir   string `json:"directory"`
	} `json:"log"`
	Port        int         `json:"port"`
	Host        string      `json:"host"`
	BaseURL     string      `json:"base_url"`
	OAuthParams OAuthParams `json:"oauth"`
	AIAPI       string      `json:"ai_api"`
	MinioParams MinioParams `json:"minio"`

	log *logrus.Logger
	mc  *minio.Client
}

type OAuthParams struct {
	ClientID     string `json:"client_id"`
	ClientSecret string `json:"client_secret"`
	JWTSecret    string `json:"jwt_secret"`
}

type MinioParams struct {
	URL       string `json:"url"`
	AccessKey string `json:"access_key"`
	SecretKey string `json:"secret_key"`
	Bucket    string `json:"bucket"`
	Prefix    string `json:"prefix"`
}

func parseLogLevel(logLevel string) (logrus.Level, error) {
	switch strings.ToLower(logLevel) {
	case "info":
		return logrus.InfoLevel, nil
	case "debug":
		return logrus.DebugLevel, nil
	case "trace":
		return logrus.TraceLevel, nil
	default:
		return logrus.InfoLevel, errors.New("level is not defined, setting info instead")
	}
}

func (c *config) Log() *logrus.Logger {
	return c.log
}

func (c *config) Minio() *minio.Client {
	return c.mc
}

func (c *config) GetMinioParams() MinioParams {
	return c.MinioParams
}

func (c *config) GetPort() int {
	return c.Port
}

func (c *config) GetHost() string {
	return c.Host
}

func (c *config) GetBaseURL() string {
	return c.BaseURL
}

func (c *config) GetOAuthParams() OAuthParams {
	return c.OAuthParams
}

func (c *config) GetAIAPI() string {
	return c.AIAPI
}
