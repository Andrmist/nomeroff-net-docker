FROM golang:alpine AS builder

WORKDIR /build

ADD go.mod .

COPY . .

RUN go mod download && go build -o autoria_rmq .

FROM alpine

WORKDIR /app

COPY --from=builder /build/west_auto_hub_numberplate /app/west_auto_hub_numberplate

CMD ["./west_auto_hub_numberplate"]
