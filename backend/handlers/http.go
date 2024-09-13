package handlers

import (
	"autoria_rmq_send/config"
	"autoria_rmq_send/utils"
	"context"
	"encoding/base64"
	"encoding/json"
	"math"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/golang-jwt/jwt"
	"github.com/sirupsen/logrus"
)

type HTTPHandler struct {
	log         *logrus.Logger
	oauthParams config.OAuthParams
	mcHandler   *MinioHandler
	aiAPI       string
	httpClient  http.Client
}

func NewHTTPHandler(log *logrus.Logger, mcHandler *MinioHandler, oauthParams config.OAuthParams, aiAPI string) HTTPHandler {
	return HTTPHandler{log: log, oauthParams: oauthParams, mcHandler: mcHandler, aiAPI: aiAPI, httpClient: http.Client{
		Timeout: 30 * time.Second,
	}}
}

// GetToken godoc
//
//	@Summary		Get OAuth 2.0 Authorization Token
//	@Description	Get [OAuth 2.0](https://www.rfc-editor.org/rfc/rfc6749.html) Authorization Token. There are two options for passing OAuth 2.0 credentials: Authorization header or x-www-form-urlencoded data
//	@Tags			est
//	@Accept			x-www-form-urlencoded
//	@Param			oauth			formData	GetTokenRequest	false	"OAuth 2.0 Client Authentication Options"
//	@Param			Authorization	header		string			false	"client_id:client_secret in base64"
//	@Produce		json
//	@Success		200	{object}	GetTokenResponce	"Returns auth token and TTL"
//	@Failure		400	{object}	utils.Http			"Request data is invalid"
//	@Failure		422	{object}	utils.Http			"Server cannot process request data"
//	@Failure		500	{object}	utils.Http
//	@Router			/oauth/token [post]
func (h *HTTPHandler) GetToken(ctx context.Context) func(c *gin.Context) {
	return func(c *gin.Context) {
		log := h.log.WithFields(logrus.Fields{
			"module": "handler_http",
			"method": "POST",
			"route":  c.FullPath(),
		})
		log.Level = h.log.Level
		var body GetTokenRequest
		if err := c.ShouldBind(&body); err != nil {
			err := utils.NewHttpError("failed to process request", err.Error(), http.StatusBadRequest)
			c.Error(err)
			return
		}
		log.Debug(body)
		if body.GrantType != "client_credentials" {
			err := utils.NewHttpError("failed to get token", "grant_type is invalid", http.StatusBadRequest)
			c.Error(err)
			return
		}

		authHeader := strings.Split(c.GetHeader("Authorization"), " ")
		if len(authHeader) == 2 {
			authHeader, err := base64.StdEncoding.DecodeString(authHeader[1])
			if err != nil {
				err := utils.NewHttpError("failed to process Authorization header", "Authorization header is invalid", http.StatusUnprocessableEntity)
				c.Error(err)
				return
			}
			clientCredentials := strings.Split(string(authHeader), ":")
			if len(clientCredentials) != 2 {
				err := utils.NewHttpError("failed to process Authorization header", "Authorization header is missing one of client credentials", http.StatusBadRequest)
				c.Error(err)
				return
			}

			if clientCredentials[0] != h.oauthParams.ClientID || clientCredentials[1] != h.oauthParams.ClientSecret {
				err := utils.NewHttpError("failed to get token", "client_id or client_secret are invalid", http.StatusBadRequest)
				c.Error(err)
				return
			}
		} else if body.ClientID != h.oauthParams.ClientID || body.ClientSecret != h.oauthParams.ClientSecret {
			err := utils.NewHttpError("failed to get token", "client_id or client_secret are invalid", http.StatusBadRequest)
			c.Error(err)
			return
		}

		ttl := 365 * 24 * time.Hour // 1 year

		token := jwt.NewWithClaims(jwt.SigningMethodHS256, &OAuthJWTClaim{
			&jwt.StandardClaims{
				ExpiresAt: time.Now().Add(ttl).Unix(),
			},
			OAuthClientRequirements{
				ClientID:     body.ClientID,
				ClientSecret: body.ClientSecret,
			},
		})
		accessToken, err := token.SignedString([]byte(h.oauthParams.JWTSecret))

		if err != nil {
			log.Error(err)
			err = utils.NewHttpError("failed to generate new token", "", http.StatusInternalServerError)
			c.Error(err)
			return
		}

		c.JSON(http.StatusOK, GetTokenResponce{
			AuthorizationToken: accessToken,
			TTL:                ttl.Milliseconds() / 1000,
		})

	}
}

// ProcessPhoto godoc
//
//	@Summary		Find numberplates on photo
//	@Description	Get numberplates array from photo file
//	@Tags			est
//	@Accept			mpfd
//	@Param			file	formData	file	true	"Numberplate photo to check. Max photo size: 100M"
//
//	@Param			Authorization	header	string				true	"OAuth 2.0 Authorization token"	default(Bearer <Add auth token here>)
//
//	@Produce		json
//	@Success		200	{object}	ProcessPhotoResponce	"Returns numberplates array"
//	@Failure		400	{object}	utils.Http				"Request data is invalid"
//	@Failure		422	{object}	utils.Http				"Server cannot process request data"
//	@Failure		401	{object}	utils.Http				"Authorize token is invalid or expired"
//	@Failure		500	{object}	utils.Http
//	@Router			/v1/numberplate [post]
func (h *HTTPHandler) ProcessPhoto(ctx context.Context) func(c *gin.Context) {
	return func(c *gin.Context) {
		log := h.log.WithFields(logrus.Fields{
			"module": "handler_http",
			"method": "POST",
			"route":  c.FullPath(),
		})
		log.Level = h.log.Level
		file, err := c.FormFile("file")
		if err != nil {
			log.Error(err)
			err = utils.NewHttpError("failed to process photo", "cannot load file", http.StatusUnprocessableEntity)
			c.Error(err)
			return
		}

		path, err := h.mcHandler.UploadFile(ctx, file)
		if err != nil {
			log.Error(err)
			err = utils.NewHttpError("failed to process photo", "cannot upload file", http.StatusUnprocessableEntity)
			c.Error(err)
			return
		}

		aiURL, err := url.Parse(h.aiAPI)
		if err != nil {
			log.Error(err)
			err = utils.NewHttpError("failed to process numberplate", "cannot parse service url", http.StatusInternalServerError)
			c.Error(err)
			return
		}

		aiURL.Path = "/read"
		query := aiURL.Query()
		query.Add("url", path)
		aiURL.RawQuery = query.Encode()
		log.Debug(aiURL.String())

		req, err := http.NewRequest(http.MethodGet, aiURL.String(), nil)
		if err != nil {
			log.Error(err)
			err = utils.NewHttpError("failed to process numberplate", "cannot init request", http.StatusInternalServerError)
			c.Error(err)
			return
		}

		res, err := h.httpClient.Do(req)
		if err != nil {
			log.Error(err)
			err = utils.NewHttpError("failed to process numberplate", "numberplate recognition request failed", http.StatusInternalServerError)
			c.Error(err)
			return
		}

		var resBody NomeroffNetResponce
		if err := json.NewDecoder(res.Body).Decode(&resBody); err != nil {
			log.Error(err)
			err = utils.NewHttpError("failed to process numberplate", "numberplate recognition request failed", http.StatusInternalServerError)
			c.Error(err)
			return
		}
		log.Debug(resBody)

		if !resBody.Validated {
			log.Error(resBody.Errors)
			err = utils.NewHttpError("failed to process numberplate", "numberplate recognition failed to validate data", http.StatusUnprocessableEntity)
			c.Error(err)
			return
		}
		if !resBody.Success {
			log.Error(resBody.Errors)
			err = utils.NewHttpError("failed to process numberplate", "numberplate recognition failed", http.StatusInternalServerError)
			c.Error(err)
			return
		}
		if len(resBody.Data) == 0 {
			log.Error(resBody.Errors)
			err = utils.NewHttpError("failed to process numberplate", "got empty responce from numberplate recognition", http.StatusInternalServerError)
			c.Error(err)
			return
		}

		resp := resBody.Data[0]
		type numberplateData struct {
			Numberplate string
			X1          float64
			Y1          float64
			X2          float64
			Y2          float64
		}
		var numberplates []numberplateData
		for i, bbox := range resp.ImagesBboxs {
			numberplates = append(numberplates, numberplateData{
				Numberplate: resp.Texts[i],
				X1:          bbox[0],
				Y1:          bbox[1],
				X2:          bbox[2],
				Y2:          bbox[3],
			})
		}

		slices.SortStableFunc(numberplates, func(a, b numberplateData) int {
			aRect := math.Abs(a.X2-a.X1) * math.Abs(a.Y2-a.Y1)
			bRect := math.Abs(b.X2-b.X1) * math.Abs(b.Y2-b.Y1)
			if aRect > bRect {
				return 1
			} else if aRect < bRect {
				return -1
			} else {
				return 0
			}
		})

		numberplatesResp := make([]string, 0)
		for _, numberplate := range numberplates {
			numberplatesResp = append(numberplatesResp, numberplate.Numberplate)
		}

		c.JSON(http.StatusOK, ProcessPhotoResponce{
			Numbeplates: numberplatesResp,
		})
	}
}

func (h *HTTPHandler) ProcessPhotoByURL(ctx context.Context) func(c *gin.Context) {
	return func(c *gin.Context) {
		log := h.log.WithFields(logrus.Fields{
			"module": "handler_http",
			"method": "POST",
			"route":  c.FullPath(),
		})
		log.Level = h.log.Level
		var path struct {
			URL string `json:"url"`
		}
		c.Bind(&path)

		aiURL, err := url.Parse(h.aiAPI)
		if err != nil {
			log.Error(err)
			err = utils.NewHttpError("failed to process numberplate", "cannot parse service url", http.StatusInternalServerError)
			c.Error(err)
			return
		}

		aiURL.Path = "/read"
		query := aiURL.Query()
		query.Add("url", path.URL)
		aiURL.RawQuery = query.Encode()
		log.Debug(aiURL.String())

		req, err := http.NewRequest(http.MethodGet, aiURL.String(), nil)
		if err != nil {
			log.Error(err)
			err = utils.NewHttpError("failed to process numberplate", "cannot init request", http.StatusInternalServerError)
			c.Error(err)
			return
		}

		res, err := h.httpClient.Do(req)
		if err != nil {
			log.Error(err)
			err = utils.NewHttpError("failed to process numberplate", "numberplate recognition request failed", http.StatusInternalServerError)
			c.Error(err)
			return
		}

		var resBody NomeroffNetResponce
		if err := json.NewDecoder(res.Body).Decode(&resBody); err != nil {
			log.Error(err)
			err = utils.NewHttpError("failed to process numberplate", "numberplate recognition request failed", http.StatusInternalServerError)
			c.Error(err)
			return
		}
		log.Debug(resBody)

		if !resBody.Validated {
			log.Error(resBody.Errors)
			err = utils.NewHttpError("failed to process numberplate", "numberplate recognition failed to validate data", http.StatusUnprocessableEntity)
			c.Error(err)
			return
		}
		if !resBody.Success {
			log.Error(resBody.Errors)
			err = utils.NewHttpError("failed to process numberplate", "numberplate recognition failed", http.StatusInternalServerError)
			c.Error(err)
			return
		}
		if len(resBody.Data) == 0 {
			log.Error(resBody.Errors)
			err = utils.NewHttpError("failed to process numberplate", "got empty responce from numberplate recognition", http.StatusInternalServerError)
			c.Error(err)
			return
		}

		resp := resBody.Data[0]
		type numberplateData struct {
			Numberplate string
			X1          float64
			Y1          float64
			X2          float64
			Y2          float64
		}
		var numberplates []numberplateData
		for i, bbox := range resp.ImagesBboxs {
			if len(resp.Texts) > i {
				numberplates = append(numberplates, numberplateData{
					Numberplate: resp.Texts[i],
					X1:          bbox[0],
					Y1:          bbox[1],
					X2:          bbox[2],
					Y2:          bbox[3],
				})
			}
		}

		slices.SortStableFunc(numberplates, func(a, b numberplateData) int {
			aRect := math.Abs(a.X2-a.X1) * math.Abs(a.Y2-a.Y1)
			bRect := math.Abs(b.X2-b.X1) * math.Abs(b.Y2-b.Y1)
			if aRect > bRect {
				return 1
			} else if aRect < bRect {
				return -1
			} else {
				return 0
			}
		})

		numberplatesResp := make([]string, 0)
		for _, numberplate := range numberplates {
			numberplatesResp = append(numberplatesResp, numberplate.Numberplate)
		}

		c.JSON(http.StatusOK, ProcessPhotoResponce{
			Numbeplates: numberplatesResp,
		})
	}
}
