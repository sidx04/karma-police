const express = require('express');
const path = require('path');
const cors = require('cors');
const OpenAI = require('openai');
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());
app.use(cors());

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY
});

const SYSTEM_PROMPT = `You are an expert system monitoring analyst for the System Eye, a comprehensive system monitoring dashboard.

AVAILABLE METRICS YOU CAN ANALYZE:
- CPU Usage: Overall percentage, per-core usage, load averages, physical/logical cores
- Memory: Total/used/available RAM, cached memory, buffers, swap usage
- Disk I/O: Read/write speeds, latency, per-device statistics
- Network: RX/TX bandwidth, packet rates, per-interface statistics, errors
- GPU: Utilization percentage, memory usage, temperature (if available)
- Processes: Top CPU/memory consumers, PIDs, process states (running/sleeping)
- ML Workload Classification: Real-time workload type detection and confidence scores

CONTEXT:
The user is viewing a live system monitoring dashboard. They can select specific widgets/graphs to share data with you. When they select graphs, you'll receive the current values, historical data points, and detailed metrics.

YOUR ROLE:
1. Be friendly and conversational - talk like you're chatting with a colleague
2. If no graphs are selected, ask them to select relevant graphs to help analyze their question
3. When graphs ARE selected, provide specific insights using the actual numbers from the data
4. Identify patterns, anomalies, or concerning trends
5. Offer actionable recommendations based on the metrics
6. Explain technical concepts clearly when needed
7. If you notice correlations between different metrics, point them out

CRITICAL - RESPONSE FORMAT:
- Write in plain conversational text, like a chat message or text
- DO NOT use markdown formatting (no **, no bullet points, no lists)
- DO NOT use special characters like asterisks, dashes, or hashtags for formatting
- Write naturally in paragraphs, just like texting a friend
- Use line breaks to separate thoughts, but keep it conversational
- Be concise but thorough - aim for 2-4 short paragraphs max
- Reference specific numbers naturally in your sentences

EXAMPLE GOOD RESPONSE:
"Your CPU is running at 45% right now with a load average of 2.4, which looks pretty normal for your 8-core system. That means you've got plenty of headroom.

I do notice your memory is at 78% though - you're using about 12GB out of 16GB total. This is getting a bit high, especially if you're running heavy applications. You might want to check what's consuming the most RAM.

Overall your system looks healthy, just keep an eye on that memory usage!"

EXAMPLE BAD RESPONSE (don't do this):
"**Analysis:**
- CPU: 45% utilization
- Load: 2.4 average
- **Recommendation:** Monitor memory"

Remember: Chat naturally, no markdown, no formatting symbols. Just friendly, helpful conversation.`;

app.post('/api/chat', async (req, res) => {
    try {
        const { message, graphContext, conversationHistory } = req.body;

        if (!process.env.OPENAI_API_KEY) {
            return res.status(500).json({
                error: 'OpenAI API key not configured. Please set OPENAI_API_KEY in .env file'
            });
        }

        const messages = [
            {
                role: 'system',
                content: SYSTEM_PROMPT
            }
        ];

        if (conversationHistory && Array.isArray(conversationHistory)) {
            const recentHistory = conversationHistory.slice(-10);
            messages.push(...recentHistory.map(msg => ({
                role: msg.role,
                content: msg.content
            })));
        }

        const userMessage = message + (graphContext || '');
        messages.push({
            role: 'user',
            content: userMessage
        });

        const completion = await openai.chat.completions.create({
            model: 'gpt-4o-mini',
            messages: messages,
            temperature: 0.7,
            max_tokens: 1000
        });

        const assistantMessage = completion.choices[0]?.message?.content;

        if (!assistantMessage) {
            throw new Error('No response from OpenAI');
        }

        res.json({
            response: assistantMessage
        });

    } catch (error) {
        console.error('Chat API Error:', error);

        if (error.status === 401) {
            return res.status(401).json({
                error: 'Invalid OpenAI API key'
            });
        }

        if (error.status === 429) {
            return res.status(429).json({
                error: 'Rate limit exceeded. Please try again later.'
            });
        }

        if (error.code === 'insufficient_quota') {
            return res.status(402).json({
                error: 'OpenAI API quota exceeded. Please check your billing.'
            });
        }

        res.status(500).json({
            error: error.message || 'Failed to process chat request'
        });
    }
});

app.get('/api/health', (req, res) => {
    res.json({
        status: 'healthy',
        service: 'system-eye-chat',
        timestamp: new Date().toISOString()
    });
});

app.listen(PORT, () => {
    console.log(`\n💬 System Eye's Chat Server Running`);
    console.log(`📡 Chat API: http://localhost:${PORT}/api/chat`);
    console.log(`🔑 OpenAI API Key: ${process.env.OPENAI_API_KEY ? 'Configured ✓' : 'NOT CONFIGURED ✗'}`);
    console.log(`\n✨ This server only handles AI chat`);
    console.log(`📊 Run the Go server on port 8080 for metrics and UI\n`);

    if (!process.env.OPENAI_API_KEY) {
        console.warn('⚠️  WARNING: OPENAI_API_KEY not set!');
        console.warn('   Set it in .env file or export OPENAI_API_KEY=your-key\n');
    }
});
