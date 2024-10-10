package models

import (
	"github.com/golang-jwt/jwt"
)

type GetTokenRequest struct {
	GrantType    string `form:"grant_type" validate:"required" default:"client_credentials"`
	ClientID     string `form:"client_id"`
	ClientSecret string `form:"client_secret"`
}

type GetTokenResponce struct {
	AuthorizationToken string `json:"authorization_token"`
	TTL                int64  `json:"ttl"`
}

type OAuthClientRequirements struct {
	ClientID     string `json:"client_id"`
	ClientSecret string `json:"client_secret"`
}

type OAuthJWTClaim struct {
	*jwt.StandardClaims
	OAuthClientRequirements
}

type ProcessPhotoResponce struct {
	Numberplates []string `json:"numberplates"`
}

type ProcessPhotoResponceSingle struct {
	Numberplate string `json:"numberplate"`
}

type NomeroffNetResponce struct {
	Data      []NomeroffNetResponceData `json:"data"`
	Success   bool                      `json:"success"`
	Validated bool                      `json:"validated"`
	Errors    string                    `json:"errors"`
}

type NomeroffNetResponceData struct {
	Confidences [][]float64 `json:"confidences"`
	CountLines  []int       `json:"count_lines"`
	ImagesBboxs [][]float64 `json:"images_bboxs"`
	RegionIds   []int       `json:"region_ids"`
	RegionNames []string    `json:"region_names"`
	Texts       []string    `json:"texts"`
	URL         string      `json:"url"`
}
