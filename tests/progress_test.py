"""
Unit tests for extraction/progress.py module.

Tests progress bar utilities and tqdm integration.
"""
import unittest
from unittest.mock import patch, MagicMock, Mock
import concurrent.futures
import time


class TestProgressUtilities(unittest.TestCase):
    """Test cases for progress utility functions."""

    @patch('extraction.progress.tqdm')
    def test_progress_bar(self, mock_tqdm):
        """Test progress_bar function."""
        from extraction.progress import progress_bar
        
        mock_instance = Mock()
        mock_tqdm.return_value = mock_instance
        
        result = progress_bar(total=100, desc="Test", unit="item", colour="blue")
        
        mock_tqdm.assert_called_once_with(
            total=100,
            desc="Test",
            unit="item",
            colour="blue",
            dynamic_ncols=True,
            leave=True
        )
        self.assertEqual(result, mock_instance)

    @patch('extraction.progress.tqdm')
    def test_progress_bar_default_params(self, mock_tqdm):
        """Test progress_bar with default parameters."""
        from extraction.progress import progress_bar
        
        progress_bar(total=50)
        
        mock_tqdm.assert_called_once_with(
            total=50,
            desc=None,
            unit="it",
            colour=None,
            dynamic_ncols=True,
            leave=True
        )

    @patch('extraction.progress.tqdm')
    def test_progress_iterable(self, mock_tqdm):
        """Test progress_iterable function."""
        from extraction.progress import progress_iterable
        
        mock_instance = Mock()
        mock_tqdm.return_value = mock_instance
        
        test_iterable = [1, 2, 3, 4, 5]
        result = progress_iterable(test_iterable, desc="Processing", unit="item")
        
        mock_tqdm.assert_called_once_with(
            test_iterable,
            desc="Processing",
            unit="item",
            colour=None,
            dynamic_ncols=True,
            leave=True
        )
        self.assertEqual(result, mock_instance)

    @patch('extraction.progress.tqdm')
    def test_progress_iterable_with_colour(self, mock_tqdm):
        """Test progress_iterable with colour parameter."""
        from extraction.progress import progress_iterable
        
        test_iterable = [1, 2, 3]
        progress_iterable(test_iterable, colour="green")
        
        mock_tqdm.assert_called_once_with(
            test_iterable,
            desc=None,
            unit="it",
            colour="green",
            dynamic_ncols=True,
            leave=True
        )


class TestAsCompletedWithTqdm(unittest.TestCase):
    """Test cases for as_completed_with_tqdm function."""

    @patch('extraction.progress.tqdm')
    @patch('concurrent.futures.as_completed')
    def test_as_completed_with_tqdm(self, mock_as_completed, mock_tqdm):
        """Test as_completed_with_tqdm function."""
        from extraction.progress import as_completed_with_tqdm
        
        # Mock futures and their completion
        mock_futures = [Mock(), Mock(), Mock()]
        mock_as_completed.return_value = iter(mock_futures)
        
        mock_pbar = Mock()
        mock_tqdm.return_value.__enter__ = Mock(return_value=mock_pbar)
        mock_tqdm.return_value.__exit__ = Mock(return_value=None)
        
        # Test the function
        result_list = list(as_completed_with_tqdm(
            mock_futures, 
            total=3, 
            desc="Processing", 
            unit="task",
            colour="yellow"
        ))
        
        # Verify tqdm was called correctly
        mock_tqdm.assert_called_once_with(
            total=3,
            desc="Processing",
            unit="task",
            colour="yellow",
            dynamic_ncols=True,
            leave=True
        )
        
        # Verify as_completed was called
        mock_as_completed.assert_called_once_with(mock_futures)
        
        # Verify results
        self.assertEqual(result_list, mock_futures)
        
        # Verify progress bar was updated for each future
        self.assertEqual(mock_pbar.update.call_count, 3)

    @patch('extraction.progress.tqdm')
    @patch('concurrent.futures.as_completed')
    def test_as_completed_with_tqdm_default_params(self, mock_as_completed, mock_tqdm):
        """Test as_completed_with_tqdm with default parameters."""
        from extraction.progress import as_completed_with_tqdm
        
        mock_futures = [Mock()]
        mock_as_completed.return_value = iter(mock_futures)
        
        mock_pbar = Mock()
        mock_tqdm.return_value.__enter__ = Mock(return_value=mock_pbar)
        mock_tqdm.return_value.__exit__ = Mock(return_value=None)
        
        list(as_completed_with_tqdm(mock_futures, total=1))
        
        mock_tqdm.assert_called_once_with(
            total=1,
            desc=None,
            unit="it",
            colour=None,
            dynamic_ncols=True,
            leave=True
        )


class TestTqdmLoggingHandler(unittest.TestCase):
    """Test cases for TqdmLoggingHandler."""

    def test_tqdm_logging_handler_exists(self):
        """Test that TqdmLoggingHandler class exists."""
        from extraction.progress import TqdmLoggingHandler
        
        # Should be able to import and instantiate
        handler = TqdmLoggingHandler()
        self.assertIsNotNone(handler)

    @patch('extraction.progress.tqdm')
    def test_tqdm_logging_handler_emit(self, mock_tqdm):
        """Test TqdmLoggingHandler emit method."""
        from extraction.progress import TqdmLoggingHandler
        import logging
        
        # Mock tqdm.write
        mock_tqdm.write = Mock()
        
        handler = TqdmLoggingHandler()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        handler.emit(record)
        
        # Should call tqdm.write with formatted message
        mock_tqdm.write.assert_called_once()
        args = mock_tqdm.write.call_args[0]
        self.assertIn("Test message", args[0])


class TestPresetProgressBars(unittest.TestCase):
    """Test cases for preset progress bar functions."""

    @patch('extraction.progress.progress_bar')
    def test_pages_progress(self, mock_progress_bar):
        """Test pages_progress preset function."""
        from extraction.progress import pages_progress
        
        mock_papers = [{"id": 1}, {"id": 2}, {"id": 3}]
        mock_progress_bar.return_value = Mock()
        
        result = pages_progress(mock_papers)
        
        mock_progress_bar.assert_called_once_with(
            total=3,
            desc="Fetching pages",
            unit="paper",
            colour="cyan"
        )
        self.assertEqual(result, mock_progress_bar.return_value)

    @patch('extraction.progress.progress_bar')
    def test_llm_progress(self, mock_progress_bar):
        """Test llm_progress preset function."""
        from extraction.progress import llm_progress
        
        mock_progress_bar.return_value = Mock()
        
        result = llm_progress(10)
        
        mock_progress_bar.assert_called_once_with(
            total=10,
            desc="LLM generate",
            unit="item",
            colour="magenta"
        )
        self.assertEqual(result, mock_progress_bar.return_value)

    def test_pages_progress_with_empty_list(self):
        """Test pages_progress with empty papers list."""
        from extraction.progress import pages_progress
        
        with patch('extraction.progress.progress_bar') as mock_progress_bar:
            pages_progress([])
            mock_progress_bar.assert_called_once_with(
                total=0,
                desc="Fetching pages",
                unit="paper",
                colour="cyan"
            )

    def test_llm_progress_with_zero_items(self):
        """Test llm_progress with zero items."""
        from extraction.progress import llm_progress
        
        with patch('extraction.progress.progress_bar') as mock_progress_bar:
            llm_progress(0)
            mock_progress_bar.assert_called_once_with(
                total=0,
                desc="LLM generate",
                unit="item",
                colour="magenta"
            )


if __name__ == '__main__':
    unittest.main()
