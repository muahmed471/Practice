# Stage 1: Build
FROM golang:1.24 AS builder

WORKDIR /app

COPY go.mod ./
RUN go mod download

COPY . .

# ✅ Disable CGO to make static binary compatible with Alpine
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o main .

# Stage 2: Minimal runtime
FROM alpine:3.20

WORKDIR /app
COPY --from=builder /app/main .

# Ensure binary is executable
RUN chmod +x /app/main

EXPOSE 8080
ENTRYPOINT ["./main"]
