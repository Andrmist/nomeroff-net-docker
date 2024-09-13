package main

import (
	"autoria_rmq_send/config"
	"autoria_rmq_send/handlers"
	"autoria_rmq_send/workers"
	"context"
	"os"
	"os/signal"
	"sync"
	"syscall"
)

const cfgPath = "./config.json"

//	@title			AutomotoAPI | WestAutoHub
//	@version		1.0
//	@description	Numberplates recognition REST API

//	@contact.name	Automoto
//	@contact.url	https://automoto.ua
//	@contact.email	info@automoto.ua

// @host		localhost:8080
// @BasePath	/api
func main() {
	cfg, err := config.Init(cfgPath)
	if err != nil {
		panic(err)
	}
	log := cfg.Log()

	ctx, cancelF := context.WithCancel(context.Background())

	mcHandler := handlers.NewMinioHandler(cfg.Minio(), cfg.GetMinioParams())

	var wg sync.WaitGroup
	go func() {
		// http worker
		wg.Add(1)
		go func() {
			defer wg.Done()
			workers.HTTPWorker(ctx, log, mcHandler, cfg.GetPort(), cfg.GetOAuthParams(), cfg.GetHost(), cfg.GetBaseURL(), cfg.GetAIAPI())
		}()
	}()

	exit := make(chan os.Signal)
	signal.Notify(exit, syscall.SIGINT, syscall.SIGTERM)

	<-exit
	log.Info("Gracefully shutting down...")

	cancelF()
	wg.Wait()

	log.Info("Shutdown")
}
