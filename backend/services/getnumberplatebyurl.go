package services

import (
	"autoria_rmq_send/models"
	"encoding/json"
	"math"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"

	"github.com/pkg/errors"
)

type numberplateData struct {
	Numberplate string
	X1          float64
	Y1          float64
	X2          float64
	Y2          float64
}

func GetNumberplateByURLs(apiURL string, urls []string) ([][]numberplateData, error) {
	client := http.Client{
		Timeout: 30 * time.Second,
	}

	aiURL, err := url.Parse(apiURL)
	if err != nil {
		return [][]numberplateData{}, errors.Wrap(err, "cannot parse url")
	}

	path := strings.Join(urls, ",")

	aiURL.Path = "/read"
	query := aiURL.Query()
	query.Add("url", path)
	aiURL.RawQuery = query.Encode()

	req, err := http.NewRequest(http.MethodGet, aiURL.String(), nil)
	if err != nil {
		return [][]numberplateData{}, errors.Wrap(err, "cannot init request")
	}

	res, err := client.Do(req)
	if err != nil {
		return [][]numberplateData{}, errors.Wrap(err, "numberplate recognition request failed")
	}

	var resBody models.NomeroffNetResponce
	if err := json.NewDecoder(res.Body).Decode(&resBody); err != nil {
		return [][]numberplateData{}, errors.Wrap(err, "numberplate recognition request failed")
	}

	if !resBody.Validated {
		return [][]numberplateData{}, errors.Wrap(err, "numberplate recognition failed to validate data")
	}
	if !resBody.Success {
		return [][]numberplateData{}, errors.Wrap(err, "numberplate recognition failed")
	}
	if len(resBody.Data) == 0 {
		return [][]numberplateData{}, errors.Wrap(err, "got empty responce from numberplate recognition")
	}

	var numberplates [][]numberplateData
	for _, resp := range resBody.Data {
		var numberplatesForPhoto []numberplateData
		for i, bbox := range resp.ImagesBboxs {
			numberplatesForPhoto = append(numberplatesForPhoto, numberplateData{
				Numberplate: resp.Texts[i],
				X1:          bbox[0],
				Y1:          bbox[1],
				X2:          bbox[2],
				Y2:          bbox[3],
			})
		}
		numberplates = append(numberplates, numberplatesForPhoto)
	}
	return numberplates, nil

	// slices.SortStableFunc(numberplates, func(a, b numberplateData) int {
	// 	aRect := math.Abs(a.X2-a.X1) * math.Abs(a.Y2-a.Y1)
	// 	bRect := math.Abs(b.X2-b.X1) * math.Abs(b.Y2-b.Y1)
	// 	if aRect > bRect {
	// 		return 1
	// 	} else if aRect < bRect {
	// 		return -1
	// 	} else {
	// 		return 0
	// 	}
	// })
}

func GetSortedNumberplatesByOnePhoto(apiURL string, url string) ([]string, error) {
	numberplates, err := GetNumberplateByURLs(apiURL, []string{url})

	if err != nil {
		return []string{}, err
	}

	slices.SortStableFunc(numberplates[0], func(a, b numberplateData) int {
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

	for _, numberplate := range numberplates[0] {
		numberplatesResp = append(numberplatesResp, numberplate.Numberplate)
	}
	return numberplatesResp, nil
}

func GetTheMostOccuredTheBiggestNumberplate(apiURL string, urls []string) (string, error) {
	numberplates, err := GetNumberplateByURLs(apiURL, urls)

	if err != nil {
		return "", err
	}

	numberplateOccurences := make(map[string]int)

	for _, numberplatesForPhoto := range numberplates {
		slices.SortStableFunc(numberplatesForPhoto, func(a, b numberplateData) int {
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
		if len(numberplatesForPhoto) > 0 {
			if _, ok := numberplateOccurences[numberplatesForPhoto[0].Numberplate]; !ok {
				numberplateOccurences[numberplatesForPhoto[0].Numberplate] = 0
			}
			numberplateOccurences[numberplatesForPhoto[0].Numberplate]++
		}
	}

	var maxNumberplate string
	for maxNumberplate = range numberplateOccurences {
		break
	}
	for numberplate, count := range numberplateOccurences {
		if count > numberplateOccurences[maxNumberplate] {
			maxNumberplate = numberplate
		}
	}

	return maxNumberplate, nil
}
