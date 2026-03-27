"""
Transcript Analyzer Agent.

Analyzes meeting transcripts to extract structured information
including participants, topics discussed, sentiment, and key moments.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.core.base_agent import (
    BaseAgent,
    AgentCapability,
    AgentConfig,
    AgentContext,
    AgentResult,
)
from src.core.audit import AuditLevel, AuditCategory


class Speaker(BaseModel):
    """Identified speaker in the transcript."""
    name: str
    email: Optional[str] = None
    speaking_time_seconds: float = 0.0
    turn_count: int = 0
    sentiment_score: float = 0.0  # -1 to 1


class Topic(BaseModel):
    """Topic discussed during the meeting."""
    name: str
    description: str
    start_time: Optional[str] = None
    duration_seconds: float = 0.0
    participants: List[str] = Field(default_factory=list)
    sentiment: str = "neutral"  # positive, negative, neutral


class TranscriptSegment(BaseModel):
    """A segment of the transcript."""
    speaker: str
    text: str
    timestamp: Optional[str] = None
    duration_seconds: float = 0.0
    sentiment: str = "neutral"


class TranscriptAnalysis(BaseModel):
    """Complete analysis of a meeting transcript."""
    meeting_title: Optional[str] = None
    meeting_date: Optional[datetime] = None
    duration_minutes: float = 0.0
    
    speakers: List[Speaker] = Field(default_factory=list)
    topics: List[Topic] = Field(default_factory=list)
    segments: List[TranscriptSegment] = Field(default_factory=list)
    
    summary: str = ""
    key_points: List[str] = Field(default_factory=list)
    overall_sentiment: str = "neutral"
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TranscriptInput(BaseModel):
    """Input for transcript analysis."""
    transcript_text: str
    meeting_title: Optional[str] = None
    meeting_date: Optional[datetime] = None
    known_participants: List[str] = Field(default_factory=list)
    language: str = "en"


class TranscriptAnalyzer(BaseAgent[TranscriptInput]):
    """
    Agent that analyzes meeting transcripts to extract structured information.
    
    Capabilities:
    - Speaker identification and diarization
    - Topic segmentation
    - Sentiment analysis
    - Key point extraction
    - Meeting summarization
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(
            name="TranscriptAnalyzer",
            capabilities=[AgentCapability.EXTRACTION],
            config=config or AgentConfig(
                confidence_threshold=0.6,
                max_retries=2
            )
        )
        
        # Patterns for identifying speakers
        self._speaker_patterns = [
            r'^([A-Z][a-z]+(?:\s[A-Z][a-z]+)?):',  # "John Smith:"
            r'\[([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\]',  # "[John Smith]"
            r'^Speaker\s*(\d+):',  # "Speaker 1:"
        ]
        
        # Sentiment indicators
        self._positive_words = {
            'agree', 'great', 'excellent', 'good', 'yes', 'approve',
            'happy', 'success', 'achieved', 'completed', 'progress'
        }
        self._negative_words = {
            'disagree', 'problem', 'issue', 'concern', 'no', 'failed',
            'delay', 'blocker', 'stuck', 'unfortunately', 'risk'
        }
    
    async def process(
        self,
        input_data: TranscriptInput,
        context: AgentContext
    ) -> AgentResult:
        """Analyze the transcript and extract structured information."""
        
        self._log_audit_event(
            event_type="transcript_analysis_start",
            context=context,
            details={
                "transcript_length": len(input_data.transcript_text),
                "known_participants": len(input_data.known_participants)
            }
        )
        
        try:
            # Guard against empty input
            if not input_data.transcript_text or not input_data.transcript_text.strip():
                return AgentResult(success=False, error="Empty transcript text provided", confidence=0.0)

            # Parse transcript into segments
            segments = self._parse_segments(input_data.transcript_text)
            
            # Identify speakers
            speakers = self._identify_speakers(
                segments, 
                input_data.known_participants
            )
            
            # Extract topics
            topics = self._extract_topics(segments)
            
            # Analyze sentiment
            overall_sentiment, segments = self._analyze_sentiment(segments)
            
            # Generate summary and key points
            summary = self._generate_summary(segments, topics)
            key_points = self._extract_key_points(segments)
            
            # Calculate duration
            duration = self._estimate_duration(segments)
            
            analysis = TranscriptAnalysis(
                meeting_title=input_data.meeting_title,
                meeting_date=input_data.meeting_date,
                duration_minutes=duration,
                speakers=speakers,
                topics=topics,
                segments=segments,
                summary=summary,
                key_points=key_points,
                overall_sentiment=overall_sentiment,
                metadata={
                    "language": input_data.language,
                    "segment_count": len(segments),
                    "word_count": len(input_data.transcript_text.split())
                }
            )
            
            # Calculate confidence based on extraction quality
            confidence = self._calculate_confidence(analysis, input_data)
            
            self._log_audit_event(
                event_type="transcript_analysis_complete",
                context=context,
                details={
                    "speakers_found": len(speakers),
                    "topics_found": len(topics),
                    "key_points": len(key_points),
                    "confidence": confidence
                }
            )
            
            return AgentResult(
                success=True,
                data=analysis.model_dump(),
                confidence=confidence,
                reasoning=f"Extracted {len(speakers)} speakers, {len(topics)} topics, and {len(key_points)} key points from transcript",
                metadata={"analysis_type": "full"}
            )
            
        except Exception as e:
            self._log_audit_event(
                event_type="transcript_analysis_error",
                context=context,
                details={"error": str(e)}
            )
            return AgentResult(
                success=False,
                error=str(e),
                confidence=0.0
            )
    
    def _parse_segments(self, text: str) -> List[TranscriptSegment]:
        """Parse transcript text into speaker segments."""
        segments = []
        current_speaker = "Unknown"
        current_text = []
        
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for speaker pattern
            speaker_found = False
            for pattern in self._speaker_patterns:
                match = re.match(pattern, line)
                if match:
                    # Save previous segment
                    if current_text:
                        segments.append(TranscriptSegment(
                            speaker=current_speaker,
                            text=' '.join(current_text)
                        ))
                    
                    current_speaker = match.group(1)
                    remaining_text = line[match.end():].strip()
                    current_text = [remaining_text] if remaining_text else []
                    speaker_found = True
                    break
            
            if not speaker_found:
                current_text.append(line)
        
        # Don't forget last segment
        if current_text:
            segments.append(TranscriptSegment(
                speaker=current_speaker,
                text=' '.join(current_text)
            ))
        
        return segments
    
    def _identify_speakers(
        self,
        segments: List[TranscriptSegment],
        known_participants: List[str]
    ) -> List[Speaker]:
        """Identify and analyze speakers from segments."""
        speaker_data = {}
        
        for segment in segments:
            speaker_name = segment.speaker
            
            # Try to match with known participants
            for participant in known_participants:
                if speaker_name.lower() in participant.lower() or \
                   participant.lower() in speaker_name.lower():
                    speaker_name = participant
                    break
            
            if speaker_name not in speaker_data:
                speaker_data[speaker_name] = {
                    "turns": 0,
                    "words": 0,
                    "sentiment_sum": 0.0
                }
            
            speaker_data[speaker_name]["turns"] += 1
            speaker_data[speaker_name]["words"] += len(segment.text.split())
        
        speakers = []
        for name, data in speaker_data.items():
            # Estimate speaking time: ~150 words per minute
            speaking_time = data["words"] / 150 * 60
            speakers.append(Speaker(
                name=name,
                speaking_time_seconds=speaking_time,
                turn_count=data["turns"]
            ))
        
        return speakers
    
    def _extract_topics(self, segments: List[TranscriptSegment]) -> List[Topic]:
        """Extract topics discussed from segments."""
        topics = []
        topic_keywords = {}
        
        # Identify potential topic indicators
        topic_indicators = [
            "let's discuss", "moving on to", "regarding", "about the",
            "next item", "agenda item", "topic:", "let's talk about"
        ]
        
        current_topic = None
        current_topic_text = []
        
        for segment in segments:
            text_lower = segment.text.lower()
            
            # Check for topic transition
            for indicator in topic_indicators:
                if indicator in text_lower:
                    # Save previous topic
                    if current_topic:
                        topics.append(Topic(
                            name=current_topic,
                            description=" ".join(current_topic_text[:100])[:200],
                            participants=list(set(s.speaker for s in segments 
                                                 if any(t in s.text.lower() 
                                                       for t in current_topic.lower().split())))
                        ))
                    
                    # Extract new topic name
                    idx = text_lower.find(indicator)
                    topic_text = segment.text[idx + len(indicator):].strip()
                    words = topic_text.split()[:5]  # First 5 words
                    current_topic = ' '.join(words).strip('.,!?')
                    current_topic_text = [segment.text]
                    break
            else:
                if current_topic_text:
                    current_topic_text.append(segment.text)
        
        # Add last topic
        if current_topic:
            topics.append(Topic(
                name=current_topic,
                description=" ".join(current_topic_text[:100])[:200]
            ))
        
        # If no explicit topics found, create general topics from keywords
        if not topics:
            all_text = ' '.join(s.text for s in segments)
            # Simple keyword extraction
            words = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', all_text)
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            for word, count in top_words:
                if count > 1:
                    topics.append(Topic(
                        name=word,
                        description=f"Mentioned {count} times"
                    ))
        
        return topics
    
    def _analyze_sentiment(
        self,
        segments: List[TranscriptSegment]
    ) -> tuple[str, List[TranscriptSegment]]:
        """Analyze sentiment of segments."""
        total_sentiment = 0.0
        
        for segment in segments:
            words = set(segment.text.lower().split())
            positive = len(words & self._positive_words)
            negative = len(words & self._negative_words)
            
            if positive > negative:
                segment.sentiment = "positive"
                total_sentiment += 1
            elif negative > positive:
                segment.sentiment = "negative"
                total_sentiment -= 1
            else:
                segment.sentiment = "neutral"
        
        if total_sentiment > len(segments) * 0.1:
            overall = "positive"
        elif total_sentiment < -len(segments) * 0.1:
            overall = "negative"
        else:
            overall = "neutral"
        
        return overall, segments
    
    def _generate_summary(
        self,
        segments: List[TranscriptSegment],
        topics: List[Topic]
    ) -> str:
        """Generate a meeting summary."""
        if not segments:
            return "No content available for summary."
        
        # Extract first sentence from each speaker change
        summary_parts = []
        seen_speakers = set()
        
        for segment in segments[:10]:  # First 10 segments
            if segment.speaker not in seen_speakers:
                seen_speakers.add(segment.speaker)
                first_sentence = segment.text.split('.')[0]
                if len(first_sentence) > 20:
                    summary_parts.append(f"{segment.speaker} discussed: {first_sentence[:100]}...")
        
        topic_summary = ""
        if topics:
            topic_names = [t.name for t in topics[:5]]
            topic_summary = f" Key topics: {', '.join(topic_names)}."
        
        return f"Meeting with {len(seen_speakers)} participants.{topic_summary} " + \
               ' '.join(summary_parts[:3])
    
    def _extract_key_points(
        self,
        segments: List[TranscriptSegment]
    ) -> List[str]:
        """Extract key points from segments."""
        key_points = []
        
        key_indicators = [
            "important", "key", "critical", "must", "need to", "decision",
            "agreed", "action", "deadline", "priority", "conclusion"
        ]
        
        for segment in segments:
            text_lower = segment.text.lower()
            for indicator in key_indicators:
                if indicator in text_lower:
                    # Extract sentence containing the indicator
                    sentences = segment.text.split('.')
                    for sentence in sentences:
                        if indicator in sentence.lower() and len(sentence) > 20:
                            key_points.append(sentence.strip())
                            break
                    break
        
        # Deduplicate and limit
        seen = set()
        unique_points = []
        for point in key_points:
            point_lower = point.lower()
            if point_lower not in seen:
                seen.add(point_lower)
                unique_points.append(point)
        
        return unique_points[:10]
    
    def _estimate_duration(self, segments: List[TranscriptSegment]) -> float:
        """Estimate meeting duration in minutes."""
        total_words = sum(len(s.text.split()) for s in segments)
        # Assume ~150 words per minute speaking rate
        return total_words / 150
    
    def _calculate_confidence(
        self,
        analysis: TranscriptAnalysis,
        input_data: TranscriptInput
    ) -> float:
        """Calculate confidence score for the analysis."""
        confidence = 0.5  # Base confidence
        
        # Boost for speakers found
        if len(analysis.speakers) >= 2:
            confidence += 0.1
        
        # Boost for topics extracted
        if len(analysis.topics) >= 1:
            confidence += 0.1
        
        # Boost for key points found
        if len(analysis.key_points) >= 3:
            confidence += 0.1
        
        # Boost for sufficient content
        word_count = len(input_data.transcript_text.split())
        if word_count >= 500:
            confidence += 0.1
        
        # Boost for known participants matched
        matched_participants = sum(
            1 for s in analysis.speakers
            if any(p.lower() in s.name.lower() or s.name.lower() in p.lower()
                   for p in input_data.known_participants)
        )
        if matched_participants > 0:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    async def _attempt_self_correction(
        self,
        input_data: TranscriptInput,
        context: AgentContext,
        failed_result: AgentResult
    ) -> Optional[AgentResult]:
        """Attempt to correct analysis failures."""
        # If initial analysis failed, try with looser parsing
        if "parse" in str(failed_result.error).lower():
            self._log_audit_event(
                event_type="self_correction_attempt",
                context=context,
                details={"strategy": "loose_parsing"}
            )
            
            # Try treating entire text as single segment
            simple_analysis = TranscriptAnalysis(
                meeting_title=input_data.meeting_title,
                meeting_date=input_data.meeting_date,
                segments=[TranscriptSegment(
                    speaker="Unknown",
                    text=input_data.transcript_text
                )],
                summary=input_data.transcript_text[:500] + "...",
                metadata={"fallback_analysis": True}
            )
            
            return AgentResult(
                success=True,
                data=simple_analysis.model_dump(),
                confidence=0.3,
                reasoning="Used fallback analysis due to parsing issues",
                warnings=["Analysis used fallback mode - results may be incomplete"]
            )
        
        return None
