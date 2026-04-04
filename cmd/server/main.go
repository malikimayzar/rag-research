package main

import (
	"encoding/json"
	"log"
	"os"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/template/html/v2"
)

type RagasResult struct {
	ExpID               string  `json:"exp_id"`
	AvgFaithfulness     float64 `json:"avg_faithfulness"`
	AvgAnswerRelevancy  float64 `json:"avg_answer_relevancy"`
	AvgContextPrecision float64 `json:"avg_context_precision"`
	AvgContextRecall    float64 `json:"avg_context_recall"`
	AvgAnswerCorrectness float64 `json:"avg_answer_correctness"`
	HallucinationRate   float64 `json:"hallucination_rate"`
}

func main() {
	// 1. Inisialisasi template engine
	engine := html.New("./web/templates", ".html")

	// 2. FIX: Tambahkan fungsi "mul" supaya bisa perkalian di HTML
	engine.AddFunc("mul", func(a, b float64) float64 {
		return a * b
	})

	app := fiber.New(fiber.Config{
		Views: engine,
	})

	// Serve folder results/metrics sebagai /static
	// Supaya gambar radar_chart.png bisa dipanggil di HTML
	app.Static("/static", "./results/metrics")

	app.Get("/", func(c *fiber.Ctx) error {
		data, err := os.ReadFile("./results/metrics/ragas_results.json")
		if err != nil {
			return c.Status(500).SendString("Gagal baca data JSON. Pastikan file ada di results/metrics/ragas_results.json")
		}

		var results []RagasResult
		if err := json.Unmarshal(data, &results); err != nil {
			return c.Status(500).SendString("Format JSON rusak.")
		}

		return c.Render("index", fiber.Map{
			"Title":   "RAG Research Dashboard",
			"Results": results,
		})
	})

	log.Println("[OK] Dashboard Go jalan di http://localhost:8080")
	log.Fatal(app.Listen(":8080"))
}