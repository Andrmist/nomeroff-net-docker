package config

import (
	"encoding/json"
	"os"

	"github.com/minio/minio-go/v7"
	"github.com/sirupsen/logrus"
)

type Config interface {
	Log() *logrus.Logger
	Minio() *minio.Client

	GetPort() int
	GetOAuthParams() OAuthParams
	GetHost() string
	GetBaseURL() string
	GetMinioParams() MinioParams
	GetAIAPI() string
}

func Init(cfgPath string) (Config, error) {
	file, err := os.Open(cfgPath)
	if err != nil {
		return nil, err
	}

	var cfg config
	if err := json.NewDecoder(file).Decode(&cfg); err != nil {
		return nil, err
	}

	log := initLog(cfg.LogParams.Level, cfg.LogParams.Dir)
	mc := initMinio(cfg.MinioParams.URL, cfg.MinioParams.AccessKey, cfg.MinioParams.SecretKey)

	cfg.mc = mc
	cfg.log = log

	return &cfg, nil
}
