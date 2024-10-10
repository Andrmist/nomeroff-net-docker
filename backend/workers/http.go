package workers

import (
	"autoria_rmq_send/config"
	"autoria_rmq_send/handlers"
	"autoria_rmq_send/utils"
	"context"
	"fmt"
	"net/http"
	"time"

	docs "autoria_rmq_send/docs"

	"github.com/gin-gonic/gin"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	swaggerfiles "github.com/swaggo/files"
	ginSwagger "github.com/swaggo/gin-swagger"
)

func httpErrorHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Next()
		for _, err := range c.Errors {
			switch e := err.Err.(type) {
			case utils.Http:
				c.AbortWithStatusJSON(e.StatusCode, e)
			default:
				c.AbortWithStatusJSON(http.StatusInternalServerError, map[string]string{"message": "Service Unavailable"})
			}
		}
	}
}

func HTTPWorker(ctx context.Context, log *logrus.Logger, mcHandler *handlers.MinioHandler, port int, oauthParams config.OAuthParams, host string, baseURL string, aiAPI string) {
	h := handlers.NewHTTPHandler(log, mcHandler, oauthParams, aiAPI)
	// limiter := utils.NewHTTPLimiter()

	r := gin.Default()
	r.Use(httpErrorHandler())
	// r.Use(limiter.Handler())

	docs.SwaggerInfo.BasePath = baseURL
	docs.SwaggerInfo.Host = host
	docs.SwaggerInfo.Schemes = []string{"http", "https"}

	api := r.Group(baseURL)
	api.POST("/oauth/token", h.GetToken(ctx))
	api.POST("/numberplate/by_url", h.ProcessPhotoByURL(ctx))
	api.POST("/numberplate/by_urls", h.ProcessPhotoByURLs(ctx))
	api.GET("/swagger/*any", ginSwagger.WrapHandler(swaggerfiles.Handler))

	v1 := api.Group("/v1")
	v1.Use(utils.AuthHandler(oauthParams.JWTSecret))
	v1.POST("/numberplate", h.ProcessPhoto(ctx))

	srv := &http.Server{
		Addr:    fmt.Sprintf("0.0.0.0:%d", port),
		Handler: r.Handler(),
	}
	go func() {
		log.Infof("HTTP worker started on port %v", port)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatal(errors.Wrap(err, "http listen"))
		}
	}()

	<-ctx.Done()
	shutdownCtx := context.Background()
	shutdownCtx, cancel := context.WithTimeout(shutdownCtx, 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Fatal(errors.Wrap(err, "shutdown http"))
	}
}
