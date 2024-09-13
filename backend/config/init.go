package config

import (
	"fmt"
	"time"

	_ "github.com/lib/pq"
	"github.com/mattn/go-colorable"
	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"github.com/snowzach/rotatefilehook"
)

func initLog(logLevel string, logPath string) *logrus.Logger {
	log := logrus.New()
	lvl, err := parseLogLevel(logLevel)
	if err != nil {
		log.Warn(err)
	}
	log.SetLevel(lvl)

	if logPath != "" {
		rotateFileHook, err := rotatefilehook.NewRotateFileHook(rotatefilehook.RotateFileConfig{
			Filename:   logPath + fmt.Sprintf("/log_%v.log", time.Now().Format(time.RFC3339)),
			MaxSize:    50, // megabytes
			MaxBackups: 3,
			MaxAge:     28, //days
			Level:      log.Level,
			Formatter: &logrus.JSONFormatter{
				TimestampFormat: time.RFC822,
			},
		})
		if err != nil {
			logrus.Fatalf("Failed to initialize file rotate hook: %v", err)
		}

		log.SetLevel(log.Level)
		log.SetOutput(colorable.NewColorableStdout())
		log.SetFormatter(&logrus.TextFormatter{
			ForceColors:     true,
			FullTimestamp:   true,
			TimestampFormat: time.RFC822,
		})
		log.AddHook(rotateFileHook)
	}

	return log
}

func initMinio(url, accessKey, secretKey string) *minio.Client {
	client, err := minio.New(url, &minio.Options{
		Creds: credentials.NewStaticV4(accessKey, secretKey, ""),
	})

	if err != nil {
		panic(errors.Wrap(err, "failed to init minio"))
	}

	return client
}
