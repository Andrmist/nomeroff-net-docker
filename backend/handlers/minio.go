package handlers

import (
	"autoria_rmq_send/config"
	"context"
	"fmt"
	"mime/multipart"
	"path/filepath"

	"github.com/google/uuid"
	"github.com/minio/minio-go/v7"
)

type MinioHandler struct {
	mc     *minio.Client
	params config.MinioParams
}

func NewMinioHandler(mc *minio.Client, params config.MinioParams) *MinioHandler {
	return &MinioHandler{mc: mc, params: params}
}

func (m *MinioHandler) UploadFile(ctx context.Context, file *multipart.FileHeader) (string, error) {
	src, err := file.Open()
	if err != nil {
		return "", err
	}
	defer src.Close()

	objectID := uuid.New().String()
	resPath := filepath.Join(m.params.Prefix, objectID)
	_, err = m.mc.PutObject(ctx, m.params.Bucket, resPath, src, file.Size, minio.PutObjectOptions{})
	if err != nil {
		return "", err
	}

	return fmt.Sprintf("http://%s/%s/%s", m.params.URL, m.params.Bucket, resPath), nil
}
